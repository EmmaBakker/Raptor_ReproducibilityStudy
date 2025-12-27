#!/usr/bin/env python3
"""
It: :
  A. Baseline vs RAPTOR leaf-overlap recall.
  B. Layer / leaf-vs-summary token distribution for RAPTOR retrieval.
  C. Tree statistics from meta.json.
  D. Embedding-space statistics from cached leaf embeddings.
  E. Dump a small JSONL of worst-overlap qualitative examples.

Run e.g.:

  python analysis/raptor_analysis_full.py \
      --dataset qasper \
      --split data/processed/qasper/eval_val.jsonl \
      --seed 224 \
      --tree-dir data/raptor_trees \
      --baseline-embeds data/leaf_embeds \
      --out results/qasper_raptor_analysis.json \
      --examples-out results/qasper_raptor_examples.jsonl
"""

import argparse
import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

import faiss
import numpy as np

import tiktoken
TOK = tiktoken.get_encoding("cl100k_base")

from sbert_raptor_eval import (
    LEAF_CHUNK_TOKENS,
    RETRIEVAL_BUDGET,
    UQA_MAX_LEN,
    UQA_CONTEXT_BUDGET,
    RAPTOR_TOP_K,
    _tok_len_cl100k,
    _norm_text,
    _sha1,
    load_jsonl,
    split_text,
    build_ra_with_sbert_config,
    baseline_sbert_context_cached,
)

from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor.RetrievalAugmentation import RetrievalAugmentation
from raptor.utils import split_text

_GLOBAL_SBERT = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")



def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_doc_key(dataset: str, ex: Dict, doc_text: str) -> str:
    """
    Mirror the logic from sbert_raptor_eval.py for deriving doc_key.
    """
    if dataset == "narrativeqa":
        return ex.get("document_id") or _sha1(doc_text)
    elif dataset == "quality":
        return ex.get("article_id") or _sha1(doc_text)
    elif dataset == "qasper":
        return str(ex.get("paper_id") or ex.get("document_id") or _sha1(doc_text))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    num = float((a * b).sum())
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def sampled_pairwise_cosine_stats(
    embs: np.ndarray,
    max_pairs: int = 500,
    rng: random.Random = random,
) -> Tuple[float, float, int]:
    """
    Approximate mean/std of pairwise cosine similarity within a doc.
    """
    n = embs.shape[0]
    if n < 2:
        return 0.0, 0.0, 0
    sims = []
    for _ in range(max_pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        sims.append(cosine_sim(embs[i], embs[j]))
    if not sims:
        return 0.0, 0.0, 0
    arr = np.array(sims, dtype=np.float32)
    return float(arr.mean()), float(arr.std()), len(arr)


def baseline_retrieve_with_indices(
    leaf_embs: np.ndarray,
    leaf_texts: List[str],
    question: str,
    max_tokens: int = RETRIEVAL_BUDGET,
    top_k: int = 50,
) -> Tuple[int, List[int]]:
    """
    Reproduce SBERT baseline retrieval but also return selected leaf indices.

    Uses:
      - inner-product FAISS index on leaf_embs (NO L2 norm, to match your eval).
      - token counting with TOK.
    """
    if leaf_embs.size == 0 or not leaf_texts:
        return 0, []

    index = faiss.IndexFlatIP(leaf_embs.shape[1])
    index.add(leaf_embs.astype(np.float32))

    q_vec = np.array([_GLOBAL_SBERT.create_embedding(question)], dtype=np.float32)

    k = min(len(leaf_texts), top_k)
    _, I = index.search(q_vec, k)

    total_tokens = 0
    picked: List[int] = []
    for raw_idx in I[0]:
        idx = int(raw_idx)
        piece = leaf_texts[idx]
        piece_tokens = len(TOK.encode(piece))
        if total_tokens + piece_tokens > max_tokens:
            break
        total_tokens += piece_tokens
        picked.append(idx)

    return total_tokens, picked


def load_tree_ra(
    tree_dir: Path,
    dataset: str,
    seed: int,
    doc_key: str,
    cfg_cache: Dict[str, Any],
) -> Tuple[RetrievalAugmentation, Any]:
    """
    Load a RAPTOR tree for a given doc_key, using an existing config builder.

    Returns:
      (ra, tree)
    """
    seed_tag = f"seed{seed}"
    doc_base = tree_dir / dataset / seed_tag / doc_key
    pkl_path = doc_base.with_suffix(".pkl")

    if not pkl_path.exists():
        raise FileNotFoundError(str(pkl_path))

    # Reuse config across docs
    if "cfg" not in cfg_cache:
        cfg_cache["cfg"] = build_ra_with_sbert_config()
    cfg = cfg_cache["cfg"]

    ra = RetrievalAugmentation(cfg, tree=str(pkl_path))
    return ra, ra.tree


def load_leaf_embs(
    embed_root: Path,
    seed: int,
    doc_key: str,
    leaf_chunks: List[str],
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Load cached leaf embeddings (.npy) for a doc_key.
    If missing, optionally recompute once (heavy) – but this should not happen
    if you already ran sbert_raptor_eval.

    Returns:
      (embs, stats_dict)
    """
    seed_tag = f"seed{seed}"
    emb_path = embed_root / seed_tag / f"{doc_key}.npy"

    stats = {
        "mean_cosine": 0.0,
        "std_cosine": 0.0,
        "n_pairs": 0,
    }

    if emb_path.exists():
        embs = np.load(emb_path)
    else:
        logging.warning(f"[leaf_embeds] Missing {emb_path}, recomputing embeddings.")
        embs = np.array(
            [_GLOBAL_SBERT.create_embedding(ch) for ch in leaf_chunks],
            dtype=np.float32,
        )


    mean_c, std_c, n_pairs = sampled_pairwise_cosine_stats(embs, rng=rng)
    stats["mean_cosine"] = mean_c
    stats["std_cosine"] = std_c
    stats["n_pairs"] = n_pairs

    return embs, stats


def collect_leaf_indices(
    node_idx: int,
    tree,
    leaf_cache: Dict[int, Set[int]],
) -> Set[int]:
    """
    Recursively collect leaf indices under a given node.
    Uses memoization to avoid repeated DFS.
    """
    if node_idx in leaf_cache:
        return leaf_cache[node_idx]

    node = tree.all_nodes[node_idx]
    if not node.children:
        result = {node_idx}
    else:
        result: Set[int] = set()
        for child_idx in node.children:
            result |= collect_leaf_indices(child_idx, tree, leaf_cache)

    leaf_cache[node_idx] = result
    return result


def aggregate_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        choices=["narrativeqa", "quality", "qasper"],
        required=True,
    )
    ap.add_argument(
        "--split",
        required=True,
        help="Path to eval JSONL (e.g., .../eval_val.jsonl).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=224,
        help="Seed used in your earlier runs (tree + embeddings).",
    )
    ap.add_argument(
        "--tree-dir",
        default="data/raptor_trees",
        help="Base dir where trees were cached (as in sbert_raptor_eval).",
    )
    ap.add_argument(
        "--baseline-embeds",
        default="data/leaf_embeds",
        help="Base dir where leaf embeddings were cached.",
    )
    ap.add_argument(
        "--out",
        default="results/raptor_analysis.json",
        help="Single JSON file with aggregated stats.",
    )
    ap.add_argument(
        "--examples-out",
        default="results/raptor_examples.jsonl",
        help="JSONL file with worst-overlap qualitative examples.",
    )
    ap.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Number of worst-overlap examples to dump.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    rng = random.Random(224)  # for reproducible sampling

    dataset = args.dataset
    split_path = Path(args.split)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    tree_dir = Path(args.tree_dir)
    embed_root = Path(args.baseline_embeds)
    out_path = Path(args.out)
    examples_out_path = Path(args.examples_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    examples_out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading data from {split_path} for dataset={dataset}")
    data = load_jsonl(split_path)

    # Caches
    cfg_cache: Dict[str, Any] = {}
    doc_leaf_chunks: Dict[str, List[str]] = {}
    doc_leaf_embs: Dict[str, np.ndarray] = {}
    doc_tree: Dict[str, Any] = {}
    doc_leaf_cache: Dict[str, Dict[int, Set[int]]] = {}  # per-doc DFS memo

    # Per-doc stats (tree + embeddings)
    doc_tree_meta: Dict[str, Dict[str, Any]] = {}
    doc_embed_stats: Dict[str, Dict[str, float]] = {}

    # Per-question stats
    overlap_recalls: List[float] = []
    baseline_ctx_tokens_all: List[int] = []
    raptor_ctx_tokens_all: List[int] = []
    leaf_token_frac_all: List[float] = []
    summary_token_frac_all: List[float] = []

    # For qualitative examples
    per_question_records: List[Dict[str, Any]] = []

    def get_leaf_chunks(doc_key: str, doc_text: str) -> List[str]:
        if doc_key not in doc_leaf_chunks:
            chunks = split_text(doc_text, TOK, LEAF_CHUNK_TOKENS)
            doc_leaf_chunks[doc_key] = chunks
        return doc_leaf_chunks[doc_key]

    def get_tree(doc_key: str) -> Tuple[RetrievalAugmentation, Any]:
        if doc_key not in doc_tree:
            ra, tree = load_tree_ra(tree_dir, dataset, args.seed, doc_key, cfg_cache)
            doc_tree[doc_key] = (ra, tree)
        return doc_tree[doc_key]

    def get_leaf_embs_for_doc(doc_key: str, leaf_chunks: List[str]) -> np.ndarray:
        if doc_key not in doc_leaf_embs:
            embs, stats = load_leaf_embs(embed_root, args.seed, doc_key, leaf_chunks, rng)
            doc_leaf_embs[doc_key] = embs
            doc_embed_stats[doc_key] = stats
        return doc_leaf_embs[doc_key]

    def get_leaf_cache(doc_key: str) -> Dict[int, Set[int]]:
        if doc_key not in doc_leaf_cache:
            doc_leaf_cache[doc_key] = {}
        return doc_leaf_cache[doc_key]

    def maybe_load_tree_meta(doc_key: str):
        if doc_key in doc_tree_meta:
            return
        seed_tag = f"seed{args.seed}"
        # build a *base* path first, then use with_suffix
        if dataset == "qasper" and "." in doc_key:
            doc_key_short = doc_key.split(".")[0]
        else:
            doc_key_short = doc_key
        doc_base = tree_dir / dataset / seed_tag / doc_key_short
        meta_path = doc_base.with_suffix(".meta.json")
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                doc_tree_meta[doc_key_short] = json.load(f)


    logging.info("Starting per-question analysis ...")

    for ex_idx, ex in enumerate(data):
        doc_text = _norm_text(ex["doc_text"])
        question = _norm_text(ex["question"])
        doc_key = get_doc_key(dataset, ex, doc_text)

        leaf_chunks = get_leaf_chunks(doc_key, doc_text)
        if not leaf_chunks:
            continue

        leaf_embs = get_leaf_embs_for_doc(doc_key, leaf_chunks)

        baseline_tokens, baseline_indices = baseline_retrieve_with_indices(
            leaf_embs,
            leaf_chunks,
            question,
            max_tokens=RETRIEVAL_BUDGET,
            top_k=50,
        )
        baseline_ctx_tokens_all.append(baseline_tokens)

        try:
            ra, tree = get_tree(doc_key)
        except FileNotFoundError:
            continue

        maybe_load_tree_meta(doc_key)


        selected_nodes, raptor_context = ra.retriever.retrieve_information_collapse_tree(
            question,
            top_k=RAPTOR_TOP_K,
            max_tokens=RETRIEVAL_BUDGET,
        )

        leaf_tokens = 0
        summary_tokens = 0

        tree_leaf_indices = set(tree.leaf_nodes.keys())
        for node in selected_nodes:
            t = node.text or ""
            n_tok = len(TOK.encode(t))
            if node.index in tree_leaf_indices:
                leaf_tokens += n_tok
            else:
                summary_tokens += n_tok

        total_raptor_tokens = leaf_tokens + summary_tokens
        raptor_ctx_tokens_all.append(total_raptor_tokens)

        if total_raptor_tokens > 0:
            leaf_frac = leaf_tokens / total_raptor_tokens
            summary_frac = summary_tokens / total_raptor_tokens
            leaf_token_frac_all.append(leaf_frac)
            summary_token_frac_all.append(summary_frac)
        else:
            leaf_frac = 0.0
            summary_frac = 0.0

        baseline_leaf_set = set(baseline_indices)

        leaf_cache = get_leaf_cache(doc_key)
        raptor_leaf_set: Set[int] = set()
        for node in selected_nodes:
            raptor_leaf_set |= collect_leaf_indices(node.index, tree, leaf_cache)

        if baseline_leaf_set:
            inter = baseline_leaf_set & raptor_leaf_set
            recall = len(inter) / len(baseline_leaf_set)
            overlap_recalls.append(recall)
        else:
            recall = 0.0

        per_question_records.append(
            {
                "example_idx": ex_idx,
                "doc_key": doc_key,
                "question": question,
                "baseline_indices": baseline_indices,
                "baseline_ctx_tokens": baseline_tokens,
                "raptor_node_indices": [n.index for n in selected_nodes],
                "raptor_leaf_indices": sorted(list(raptor_leaf_set)),
                "raptor_ctx_tokens": total_raptor_tokens,
                "leaf_token_frac": leaf_frac,
                "summary_token_frac": summary_frac,
                "overlap_recall": recall,
            }
        )

    logging.info("Aggregating stats ...")

    # Overlap stats
    overlap_stats = aggregate_stats(overlap_recalls)

    # Leaf vs summary token fractions
    leaf_frac_stats = aggregate_stats(leaf_token_frac_all)
    summary_frac_stats = aggregate_stats(summary_token_frac_all)

    # Tree stats from meta.json
    tree_layers = []
    tree_num_nodes = []
    tree_num_leaves = []
    avg_cluster_sizes = []

    for doc_key, meta in doc_tree_meta.items():
        ts = meta.get("tree_stats", {})
        layers_actual = ts.get("layers_actual")
        num_nodes = ts.get("num_nodes")
        num_leaves = ts.get("num_leaves")

        if layers_actual is not None:
            tree_layers.append(layers_actual)
        if num_nodes is not None:
            tree_num_nodes.append(num_nodes)
        if num_leaves is not None:
            tree_num_leaves.append(num_leaves)
            if num_nodes and num_nodes > num_leaves:
                internal = num_nodes - num_leaves
                if internal > 0:
                    avg_cluster_sizes.append(num_leaves / internal)

    tree_stats = {
        "num_docs_with_meta": len(doc_tree_meta),
        "layers_actual": aggregate_stats(tree_layers),
        "num_nodes": aggregate_stats(tree_num_nodes),
        "num_leaves": aggregate_stats(tree_num_leaves),
        "avg_cluster_size": aggregate_stats(avg_cluster_sizes),
    }

    # Embedding stats from cached leaf embeddings
    emb_mean_cosines = []
    emb_std_cosines = []
    for stats in doc_embed_stats.values():
        if stats.get("n_pairs", 0) > 0:
            emb_mean_cosines.append(stats["mean_cosine"])
            emb_std_cosines.append(stats["std_cosine"])

    embedding_stats = {
        "num_docs_with_embs": len(doc_embed_stats),
        "intra_doc_mean_cosine": aggregate_stats(emb_mean_cosines),
        "intra_doc_std_cosine": aggregate_stats(emb_std_cosines),
    }

    # Context length stats (tokens)
    baseline_ctx_stats = aggregate_stats(baseline_ctx_tokens_all)
    raptor_ctx_stats = aggregate_stats(raptor_ctx_tokens_all)

    # Overall output
    result = {
        "dataset": dataset,
        "seed": args.seed,
        "n_examples_total": len(data),
        "n_examples_analyzed": len(per_question_records),
        "overlap_recall": overlap_stats,
        "raptor_leaf_token_frac": leaf_frac_stats,
        "raptor_summary_token_frac": summary_frac_stats,
        "baseline_context_tokens": baseline_ctx_stats,
        "raptor_context_tokens": raptor_ctx_stats,
        "tree_stats": tree_stats,
        "embedding_stats": embedding_stats,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logging.info(f"Wrote aggregated analysis to {out_path}")

    logging.info("Selecting qualitative examples ...")

    # sort by overlap_recall ascending (worst first)
    sorted_records = sorted(
        per_question_records,
        key=lambda r: r["overlap_recall"],
    )

    max_examples = max(0, int(args.max_examples))
    selected = sorted_records[:max_examples]

    # Need leaf text snippets for selected examples
    with examples_out_path.open("w", encoding="utf-8") as f:
        for rec in selected:
            doc_key = rec["doc_key"]
            doc_leafs = doc_leaf_chunks.get(doc_key, [])
            # baseline snippets: first few chunks
            baseline_snips = [
                doc_leafs[i] for i in rec["baseline_indices"][:3]
                if 0 <= i < len(doc_leafs)
            ]
            raptor_leaf_indices = rec["raptor_leaf_indices"]
            raptor_snips = [
                doc_leafs[i] for i in raptor_leaf_indices[:3]
                if 0 <= i < len(doc_leafs)
            ]

            out_rec = {
                "example_idx": rec["example_idx"],
                "doc_key": doc_key,
                "question": rec["question"],
                "overlap_recall": rec["overlap_recall"],
                "baseline_ctx_tokens": rec["baseline_ctx_tokens"],
                "raptor_ctx_tokens": rec["raptor_ctx_tokens"],
                "leaf_token_frac": rec["leaf_token_frac"],
                "summary_token_frac": rec["summary_token_frac"],
                "baseline_indices": rec["baseline_indices"],
                "raptor_node_indices": rec["raptor_node_indices"],
                "raptor_leaf_indices": raptor_leaf_indices,
                "baseline_snippets": baseline_snips,
                "raptor_snippets": raptor_snips,
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    logging.info(f"Wrote {len(selected)} qualitative examples to {examples_out_path}")


if __name__ == "__main__":
    main()
