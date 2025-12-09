#!/usr/bin/env python3
"""
For each question, compute:
  - baseline SBERT+UQA correctness
  - RAPTOR+UQA correctness
  - RAPTOR vs baseline leaf overlap recall
  - baseline and RAPTOR context token lengths (cl100k)
  - RAPTOR leaf vs summary token fractions
  - tree stats (layers / nodes / leaves) from meta.json


Assumes you have already run:
  - sbert_raptor_eval.py WITHOUT --with-raptor (for cached leaf_embeds)
  - sbert_raptor_eval.py WITH --with-raptor (for RAPTOR trees + meta)
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

import numpy as np
import faiss

from sbert_raptor_eval import (
    set_global_seed,
    LEAF_CHUNK_TOKENS,
    RETRIEVAL_BUDGET,
    UQA_MODEL_NAME,
    UQA_MAX_LEN,
    RAPTOR_TOP_K,
    TOK,
    load_jsonl,
    _norm_text,
    _sha1,
    build_ra_with_sbert_config,
    baseline_sbert_context_cached,
    clip_for_unifiedqa,
    _quality_prompt,
    _parse_quality_pred,
    RAPTORCache,
)

from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor.RetrievalAugmentation import RetrievalAugmentation
from raptor.utils import split_text
from raptor.QAModels import UnifiedQAModel


# -------------------- helpers for retrieval stats -------------------- #

_GLOBAL_SBERT = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    num = float((a * b).sum())
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def baseline_retrieve_with_indices(
    leaf_embs: np.ndarray,
    leaf_texts: List[str],
    question: str,
    max_tokens: int = RETRIEVAL_BUDGET,
    top_k: int = 50,
) -> Tuple[int, List[int]]:
    """
    SBERT baseline retrieval but also returning selected leaf indices.
    Mirrors FaissRetriever (inner-product, no L2 norm).
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


def load_leaf_embs(
    embed_root: Path,
    seed: int,
    doc_key: str,
    leaf_chunks: List[str],
) -> np.ndarray:
    """
    Load cached leaf embeddings (.npy) for a doc_key.
    If missing, recompute once (should be rare if you ran sbert_raptor_eval).
    """
    seed_tag = f"seed{seed}"
    emb_path = embed_root / seed_tag / f"{doc_key}.npy"

    if emb_path.exists():
        embs = np.load(emb_path)
    else:
        logging.warning(f"[leaf_embeds] Missing {emb_path}, recomputing embeddings.")
        embs = np.array(
            [_GLOBAL_SBERT.create_embedding(ch) for ch in leaf_chunks],
            dtype=np.float32,
        )
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embs)

    return embs


def collect_leaf_indices(
    node_idx: int,
    tree,
    leaf_cache: Dict[int, Set[int]],
) -> Set[int]:
    """Recursively collect leaf indices under a given node."""
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
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
        }
    arr = np.array(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(values),
    }


def get_doc_key(dataset: str, ex: Dict, doc_text: str) -> str:
    """
    Match sbert_raptor_eval.py: one doc_key per (dataset, document).
    """
    if dataset == "quality":
        return ex.get("article_id") or _sha1(doc_text)
    elif dataset == "narrativeqa":
        return ex.get("document_id") or _sha1(doc_text)
    elif dataset == "qasper":
        return ex.get("paper_id") or _sha1(doc_text)
    else:
        raise ValueError(f"Unknown dataset {dataset}")


# -------------------- main analysis -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["quality"], required=True,
                    help="Only 'quality' is supported here.")
    ap.add_argument("--split", required=True,
                    help="Path to eval JSONL (e.g., .../eval_val_sub50_q5.jsonl).")
    ap.add_argument("--seed", type=int, default=224)
    ap.add_argument("--tree-dir", default="data/raptor_trees",
                    help="Base dir where trees were cached.")
    ap.add_argument("--baseline-embeds", default="data/leaf_embeds",
                    help="Base dir where leaf embeddings were cached.")
    ap.add_argument("--out", default="results/quality_hard_easy_stats.json")
    ap.add_argument("--examples-out", default="results/quality_hard_easy_examples.jsonl")
    args = ap.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    set_global_seed(args.seed)
    rng = random.Random(224)

    # Paths
    split_path = Path(args.split)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    tree_root = Path(args.tree_dir)
    embed_root = Path(args.baseline_embeds)
    out_path = Path(args.out)
    examples_out_path = Path(args.examples_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    examples_out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading QuALITY data from {split_path}")
    data = load_jsonl(split_path)

    # Build RA config + cache (for loading existing trees)
    cfg = build_ra_with_sbert_config()
    seed_tag = f"seed{args.seed}"
    tree_dir = tree_root / args.dataset / seed_tag
    tree_dir.mkdir(parents=True, exist_ok=True)
    cache = RAPTORCache(cfg, tree_dir, args.seed)

    # QA model
    qa = UnifiedQAModel(UQA_MODEL_NAME)

    # Per-doc caches
    doc_leaf_chunks: Dict[str, List[str]] = {}
    doc_leaf_embs: Dict[str, np.ndarray] = {}
    doc_tree: Dict[str, Any] = {}
    doc_leaf_cache: Dict[str, Dict[int, Set[int]]] = {}
    doc_tree_meta: Dict[str, Dict[str, Any]] = {}

    def get_leaf_chunks(doc_key: str, doc_text: str) -> List[str]:
        if doc_key not in doc_leaf_chunks:
            chunks = split_text(doc_text, TOK, LEAF_CHUNK_TOKENS)
            doc_leaf_chunks[doc_key] = chunks
        return doc_leaf_chunks[doc_key]

    def get_leaf_embs_for_doc(doc_key: str, leaf_chunks: List[str]) -> np.ndarray:
        if doc_key not in doc_leaf_embs:
            doc_leaf_embs[doc_key] = load_leaf_embs(embed_root, args.seed, doc_key, leaf_chunks)
        return doc_leaf_embs[doc_key]

    def get_tree(doc_key: str) -> Tuple[RetrievalAugmentation, Any]:
        if doc_key not in doc_tree:
            # cache.get_or_build will just load if pkl already exists
            ra = cache.get_or_build("", doc_key)  # doc text unused when loading
            doc_tree[doc_key] = (ra, ra.tree)
        return doc_tree[doc_key]

    def get_leaf_cache(doc_key: str) -> Dict[int, Set[int]]:
        if doc_key not in doc_leaf_cache:
            doc_leaf_cache[doc_key] = {}
        return doc_leaf_cache[doc_key]

    def maybe_load_tree_meta(doc_key: str):
        if doc_key in doc_tree_meta:
            return
        meta_path = tree_dir / f"{doc_key}.meta.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                doc_tree_meta[doc_key] = json.load(f)

    # Containers for hard/easy subsets
    subsets = {
        "hard": {
            "overlap": [],
            "base_ctx": [],
            "raptor_ctx": [],
            "leaf_frac": [],
            "summary_frac": [],
            "base_correct": [],
            "raptor_correct": [],
            "tree_layers": [],
            "tree_nodes": [],
            "tree_leaves": [],
        },
        "easy": {
            "overlap": [],
            "base_ctx": [],
            "raptor_ctx": [],
            "leaf_frac": [],
            "summary_frac": [],
            "base_correct": [],
            "raptor_correct": [],
            "tree_layers": [],
            "tree_nodes": [],
            "tree_leaves": [],
        },
    }

    # Also dump per-question records for inspection
    examples_out = examples_out_path.open("w", encoding="utf-8")

    logging.info("Starting per-question hard/easy analysis ...")

    for ex_idx, ex in enumerate(data):
        doc = _norm_text(ex["doc_text"])
        q = _norm_text(ex["question"])
        choices = [_norm_text(c) for c in (ex.get("choices", []) or [])]
        gold_idx = ex.get("gold_idx", ex.get("correct_idx"))
        if gold_idx is None or not choices:
            continue

        is_hard = bool(ex.get("is_hard", False))
        subset_key = "hard" if is_hard else "easy"
        sub = subsets[subset_key]

        doc_key = get_doc_key(args.dataset, ex, doc)

        # --- leaf chunks + embeddings ---
        leaf_chunks = get_leaf_chunks(doc_key, doc)
        if not leaf_chunks:
            continue
        leaf_embs = get_leaf_embs_for_doc(doc_key, leaf_chunks)

        # --- baseline retrieval stats ---
        base_tokens, base_indices = baseline_retrieve_with_indices(
            leaf_embs, leaf_chunks, q, max_tokens=RETRIEVAL_BUDGET, top_k=50
        )
        sub["base_ctx"].append(base_tokens)

        # baseline context string for QA (uses cached embeddings)
        embed_cache_dir = embed_root / seed_tag
        base_ctx = baseline_sbert_context_cached(
            leaf_chunks, q, _GLOBAL_SBERT, TOK, embed_cache_dir, doc_key, RETRIEVAL_BUDGET
        )

        # --- RAPTOR retrieval + stats ---
        ra, tree = get_tree(doc_key)
        maybe_load_tree_meta(doc_key)

        selected_nodes, raptor_ctx = ra.retriever.retrieve_information_collapse_tree(
            q, top_k=RAPTOR_TOP_K, max_tokens=RETRIEVAL_BUDGET
        )

        leaf_tokens = 0
        summary_tokens = 0
        tree_leaf_indices = set(tree.leaf_nodes.keys())

        leaf_cache = get_leaf_cache(doc_key)
        raptor_leaf_set: Set[int] = set()

        for node in selected_nodes:
            t = node.text or ""
            n_tok = len(TOK.encode(t))
            if node.index in tree_leaf_indices:
                leaf_tokens += n_tok
            else:
                summary_tokens += n_tok
            # collect all leaves under this node
            raptor_leaf_set |= collect_leaf_indices(node.index, tree, leaf_cache)

        total_raptor_tokens = leaf_tokens + summary_tokens
        sub["raptor_ctx"].append(total_raptor_tokens)

        if total_raptor_tokens > 0:
            sub["leaf_frac"].append(leaf_tokens / total_raptor_tokens)
            sub["summary_frac"].append(summary_tokens / total_raptor_tokens)
        else:
            sub["leaf_frac"].append(0.0)
            sub["summary_frac"].append(0.0)

        # --- overlap recall ---
        base_leaf_set = set(base_indices)
        if base_leaf_set:
            inter = base_leaf_set & raptor_leaf_set
            recall = len(inter) / len(base_leaf_set)
        else:
            recall = 0.0
        sub["overlap"].append(recall)

        # --- tree stats per question (attach doc-level stats) ---
        meta = doc_tree_meta.get(doc_key)
        if meta is not None:
            ts = meta.get("tree_stats", {})
            layers_actual = ts.get("layers_actual")
            num_nodes = ts.get("num_nodes")
            num_leaves = ts.get("num_leaves")
            if layers_actual is not None:
                sub["tree_layers"].append(layers_actual)
            if num_nodes is not None:
                sub["tree_nodes"].append(num_nodes)
            if num_leaves is not None:
                sub["tree_leaves"].append(num_leaves)

        # --- QA: baseline + RAPTOR, same UnifiedQA model ---

        q_with_opts = _quality_prompt(q, choices)

        # Baseline
        q_trim_b, c_trim_b = clip_for_unifiedqa(q_with_opts, _norm_text(base_ctx), budget=UQA_MAX_LEN)
        pred_b = qa.answer_question(c_trim_b, q_trim_b)
        guess_b = _parse_quality_pred(pred_b, choices)
        sub["base_correct"].append(int(int(guess_b) == int(gold_idx)))

        # RAPTOR
        q_trim_r, c_trim_r = clip_for_unifiedqa(q_with_opts, _norm_text(raptor_ctx), budget=UQA_MAX_LEN)
        pred_r = qa.answer_question(c_trim_r, q_trim_r)
        guess_r = _parse_quality_pred(pred_r, choices)
        sub["raptor_correct"].append(int(int(guess_r) == int(gold_idx)))

        # --- write per-question record ---
        out_rec = {
            "example_idx": ex_idx,
            "question_id": ex.get("question_id"),
            "article_id": ex.get("article_id"),
            "is_hard": is_hard,
            "doc_key": doc_key,
            "overlap_recall": recall,
            "baseline_ctx_tokens": base_tokens,
            "raptor_ctx_tokens": total_raptor_tokens,
            "leaf_token_frac": sub["leaf_frac"][-1],
            "summary_token_frac": sub["summary_frac"][-1],
            "baseline_correct": sub["base_correct"][-1],
            "raptor_correct": sub["raptor_correct"][-1],
        }
        examples_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    examples_out.close()

    # -------------------- aggregate stats -------------------- #

    def subset_result(sub: Dict[str, List[float]]) -> Dict[str, Any]:
        n_q = len(sub["overlap"])
        base_acc = 100.0 * sum(sub["base_correct"]) / max(1, len(sub["base_correct"]))
        raptor_acc = 100.0 * sum(sub["raptor_correct"]) / max(1, len(sub["raptor_correct"]))
        return {
            "n_questions": int(n_q),
            "baseline_accuracy": float(base_acc),
            "raptor_accuracy": float(raptor_acc),
            "overlap_recall": aggregate_stats(sub["overlap"]),
            "baseline_context_tokens": aggregate_stats(sub["base_ctx"]),
            "raptor_context_tokens": aggregate_stats(sub["raptor_ctx"]),
            "leaf_token_frac": aggregate_stats(sub["leaf_frac"]),
            "summary_token_frac": aggregate_stats(sub["summary_frac"]),
            "tree_layers": aggregate_stats(sub["tree_layers"]),
            "tree_num_nodes": aggregate_stats(sub["tree_nodes"]),
            "tree_num_leaves": aggregate_stats(sub["tree_leaves"]),
        }

    result = {
        "dataset": args.dataset,
        "seed": args.seed,
        "hard": subset_result(subsets["hard"]),
        "easy": subset_result(subsets["easy"]),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logging.info(f"[OK] Wrote hard/easy stats to {out_path}")
    logging.info(f"[OK] Wrote per-question examples to {examples_out_path}")


if __name__ == "__main__":
    main()
