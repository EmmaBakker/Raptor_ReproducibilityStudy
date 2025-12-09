#!/usr/bin/env python3
"""
Analyze RAPTOR trees (UMAP / PaCMAP / TriMap) for:

1. Semantic cluster quality
   - Intra-cluster cosine similarity
   - Inter-cluster cosine similarity (cluster centroids)

2. Summary quality & coverage
   - Cosine similarity between summary node and its children
   - Compression ratio: summary_tokens / children_tokens

3. Retrieval layer distribution (collapsed-tree)
   - For a subset of questions, which layers are selected.

Usage examples (from repo root):

  python analysis/raptor_pacmap_analysis.py \
      --dataset quality \
      --trees-root data/raptor_trees \
      --seed 224 \
      --split data/processed/quality/eval_val_sub50_q5.jsonl \
      --max-examples 200 \
      --out results/analysis_quality_seed224_umap.json
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tiktoken

from raptor.tree_structures import Tree, Node
from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig
from raptor.EmbeddingModels import SBertEmbeddingModel

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                    level=logging.INFO)

# ---------- Basic IO helpers (copied from eval script style) ----------

def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _norm_text(x) -> str:
    if isinstance(x, dict):
        for k in ("text", "question", "q", "context", "doc", "document"):
            if k in x and isinstance(x[k], str):
                return x[k]
        for v in x.values():
            if isinstance(v, str):
                return v
        return str(x)
    return str(x)


def _sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def doc_key_for_example(dataset: str, ex: Dict, doc_text: str) -> str:
    if dataset == "narrativeqa":
        return ex.get("document_id") or _sha1(doc_text)
    elif dataset == "quality":
        return ex.get("article_id") or _sha1(doc_text)
    elif dataset == "qasper":
        doc_key = ex.get("paper_id") or _sha1(doc_text)
        if "." in doc_key:
            doc_key = doc_key.split(".")[0]
        return doc_key
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------- Cosine helpers ----------

def _as_np(v):
    v = np.asarray(v, dtype=np.float32)
    return v


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """
    Cosine similarity for all pairs in X (n, d).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    return Xn @ Xn.T  # (n, n)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = _as_np(a)
    b = _as_np(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


# ---------- Analysis over clusters / summaries (tree only) ----------

TOK = tiktoken.get_encoding("cl100k_base")


def analyze_tree_clusters_and_summaries(
    tree: Tree,
    embedding_key: str = "SBERT",
) -> Dict:
    """
    For a Tree, compute:
      - intra-cluster similarity (per parent, over children)
      - inter-cluster similarity (per layer, between cluster centroids)
      - summary coverage: parent-child similarity
      - compression: summary_tokens / children_tokens
    Returns aggregated stats across all layers of this tree.
    """
    intra_sims: List[float] = []
    inter_sims: List[float] = []
    parent_child_sims: List[float] = []
    compression_ratios: List[float] = []
    children_token_counts: List[int] = []
    summary_token_counts: List[int] = []

    # Iterate over non-leaf layers (parents live in layers 1..num_layers)
    for layer, nodes in tree.layer_to_nodes.items():
        if layer == 0:
            continue  # leaves only; parents are above

        # Each parent defines a cluster = its direct children
        clusters_embeddings: List[np.ndarray] = []
        cluster_centroids: List[np.ndarray] = []

        for parent in nodes:
            if not parent.children:
                continue

            child_nodes: List[Node] = [tree.all_nodes[idx] for idx in parent.children]
            if len(child_nodes) < 1:
                continue

            # Child embeddings
            child_embs = np.stack(
                [_as_np(ch.embeddings[embedding_key]) for ch in child_nodes],
                axis=0,
            )
            clusters_embeddings.append(child_embs)

            # --- Intra-cluster similarity ---
            if len(child_nodes) >= 2:
                S = cosine_sim_matrix(child_embs)
                # Take upper triangle without diagonal
                n = S.shape[0]
                vals = S[np.triu_indices(n, k=1)]
                if len(vals) > 0:
                    intra_sims.append(float(vals.mean()))

            # --- Parent-child similarity (coverage) ---
            parent_emb = _as_np(parent.embeddings[embedding_key])
            sims_pc = [cosine_sim(parent_emb, e) for e in child_embs]
            parent_child_sims.extend(sims_pc)

            # --- Compression ratio ---
            child_text = " ".join(ch.text for ch in child_nodes)
            child_tokens = len(TOK.encode(child_text))
            summary_tokens = len(TOK.encode(parent.text))

            if child_tokens > 0:
                compression_ratios.append(summary_tokens / child_tokens)
                children_token_counts.append(child_tokens)
                summary_token_counts.append(summary_tokens)

            # centroid for this cluster
            cluster_centroids.append(child_embs.mean(axis=0))

        # --- Inter-cluster similarity for this layer ---
        if len(cluster_centroids) >= 2:
            C = np.stack(cluster_centroids, axis=0)
            S_layer = cosine_sim_matrix(C)
            k = S_layer.shape[0]
            vals = S_layer[np.triu_indices(k, k=1)]
            if len(vals) > 0:
                inter_sims.append(float(vals.mean()))

    return {
        "intra_cluster_sim_mean": float(np.mean(intra_sims)) if intra_sims else None,
        "intra_cluster_sim_std": float(np.std(intra_sims)) if intra_sims else None,
        "inter_cluster_sim_mean": float(np.mean(inter_sims)) if inter_sims else None,
        "inter_cluster_sim_std": float(np.std(inter_sims)) if inter_sims else None,
        "parent_child_sim_mean": float(np.mean(parent_child_sims)) if parent_child_sims else None,
        "parent_child_sim_std": float(np.std(parent_child_sims)) if parent_child_sims else None,
        "compression_ratio_mean": float(np.mean(compression_ratios)) if compression_ratios else None,
        "compression_ratio_std": float(np.std(compression_ratios)) if compression_ratios else None,
        "children_tokens_mean": float(np.mean(children_token_counts)) if children_token_counts else None,
        "summary_tokens_mean": float(np.mean(summary_token_counts)) if summary_token_counts else None,
    }


def aggregate_tree_metrics(metrics_list: List[Dict]) -> Dict:
    """
    Aggregate per-tree metrics into dataset-level means.
    For each scalar field, average over trees that have non-None values.
    """
    if not metrics_list:
        return {}

    agg: Dict[str, float] = {}
    keys = metrics_list[0].keys()
    for k in keys:
        vals = [m[k] for m in metrics_list if m.get(k) is not None]
        if not vals:
            agg[k] = None
        else:
            arr = np.array(vals, dtype=np.float32)
            agg[k + "_across_docs_mean"] = float(arr.mean())
            agg[k + "_across_docs_std"] = float(arr.std())
    return agg


# ---------- Retrieval layer distribution ----------

def build_tree_retriever_for_tree(tree: Tree) -> TreeRetriever:
    """
    Construct a TreeRetriever instance for a given tree, using SBERT like in sbert_raptor_eval.
    """
    sbert = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")
    cfg = TreeRetrieverConfig(
        tokenizer=TOK,
        threshold=0.5,
        top_k=50,
        selection_mode="top_k",
        context_embedding_model="SBERT",
        embedding_model=sbert,
        num_layers=None,
        start_layer=None,
    )
    retriever = TreeRetriever(cfg, tree)
    return retriever


def analyze_retrieval_layer_distribution(
    dataset: str,
    split_path: Path,
    tree_dir: Path,
    max_examples: int = 200,
) -> Dict:
    """
    For up to max_examples QA pairs:
      - Load tree for the doc
      - Run collapsed-tree retrieval
      - Record which layers are selected
    Returns: {layer_number: count, ...} plus normalized distribution.
    """
    data = load_jsonl(split_path)
    layer_counts: Dict[int, int] = {}
    n_used = 0

    for ex in data:
        if n_used >= max_examples:
            break

        doc = _norm_text(ex["doc_text"])
        q = _norm_text(ex["question"])
        doc_key = doc_key_for_example(dataset, ex, doc)

        pkl_path = tree_dir / f"{doc_key}.pkl"
        if not pkl_path.exists():
            # No tree for this doc (e.g. run crashed mid-way)
            continue

        with open(pkl_path, "rb") as f:
            tree: Tree = pickle.load(f)

        retriever = build_tree_retriever_for_tree(tree)

        # collapsed_tree retrieval; just need layer info
        _, layer_info = retriever.retrieve(
            q,
            top_k=50,
            max_tokens=2000,
            collapse_tree=True,
            return_layer_information=True,
        )

        for info in layer_info:
            layer = int(info["layer_number"])
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        n_used += 1

    total_selected = sum(layer_counts.values()) or 1
    layer_dist = {
        str(layer): {
            "count": count,
            "fraction": count / total_selected,
        }
        for layer, count in sorted(layer_counts.items(), key=lambda x: x[0])
    }

    return {
        "num_questions_used": n_used,
        "total_selected_nodes": total_selected,
        "layer_distribution": layer_dist,
    }


# ---------- Main driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["narrativeqa", "quality", "qasper"], required=True)
    ap.add_argument("--trees-root", default="data/raptor_trees",
                    help="Root dir containing per-dataset/seed trees.")
    ap.add_argument("--seed", type=int, required=True,
                    help="Seed tag used in raptor_trees (e.g. 224 for UMAP, 42 for PaCMAP, etc.).")
    ap.add_argument("--split", required=True,
                    help="Path to eval JSONL (same split you used for evaluation).")
    ap.add_argument("--max-examples", type=int, default=200,
                    help="Max QA pairs for retrieval-layer analysis.")
    ap.add_argument("--out", required=True,
                    help="Path to JSON output with analysis results.")
    args = ap.parse_args()

    dataset = args.dataset
    trees_root = Path(args.trees_root)
    possible_dirs = [
        trees_root / dataset / f"seed{args.seed}",
        trees_root / dataset / f"seed{args.seed}_trimap",
        trees_root / dataset / f"seed{args.seed}_pacmap",
        trees_root / dataset / f"seed{args.seed}_umap",
    ]

    tree_dir = None
    for d in possible_dirs:
        if d.exists():
            tree_dir = d
            break

    if tree_dir is None:
        raise FileNotFoundError(
            f"No tree directory found for seed={args.seed}. "
            f"Tried: {[str(d) for d in possible_dirs]}"
        )

    logging.info(f"Using tree directory: {tree_dir}")

    split_path = Path(args.split)
    out_path = Path(args.out)

    if not tree_dir.exists():
        raise FileNotFoundError(f"Tree dir not found: {tree_dir.resolve()}")

    logging.info(f"Analyzing trees in: {tree_dir}")
    logging.info(f"Dataset: {dataset}, seed: {args.seed}")
    logging.info(f"Split for retrieval layer analysis: {split_path}")

    # --- 1 & 2: Cluster quality + summary coverage ---
    per_tree_metrics: List[Dict] = []
    num_trees = 0
    dr_methods_found = set()

    for pkl_path in tree_dir.glob("*.pkl"):
        try:
            with open(pkl_path, "rb") as f:
                tree: Tree = pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to load tree {pkl_path}: {e}")
            continue

        # Try to read dr_method from the corresponding meta.json
        meta_path = pkl_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                cp = meta.get("clustering_params", {}) or {}
                dm = cp.get("dr_method")
                if dm is not None:
                    dr_methods_found.add(dm)
            except Exception as e:
                logging.warning(f"Failed to read meta for {pkl_path}: {e}")

        m = analyze_tree_clusters_and_summaries(tree, embedding_key="SBERT")
        per_tree_metrics.append(m)
        num_trees += 1

    logging.info(f"Loaded and analyzed {num_trees} trees for cluster/summary metrics.")

    agg_metrics = aggregate_tree_metrics(per_tree_metrics)

    # Determine dr_method (if consistent)
    if not dr_methods_found:
        dr_method = None
    elif len(dr_methods_found) == 1:
        dr_method = list(dr_methods_found)[0]
    else:
        dr_method = sorted(list(dr_methods_found))
        logging.warning(f"Multiple dr_methods found in {tree_dir}: {dr_method}")

    # --- 3: Retrieval layer distribution ---
    retrieval_layers = analyze_retrieval_layer_distribution(
        dataset, split_path, tree_dir, max_examples=args.max_examples
    )

    result = {
        "dataset": dataset,
        "seed": args.seed,
        "trees_root": str(trees_root),
        "tree_dir": str(tree_dir),
        "dr_method": dr_method,  # <-- UMAP / PaCMAP / TriMap (or list/missing)
        "num_trees_analyzed": num_trees,
        "cluster_and_summary_metrics": agg_metrics,
        "retrieval_layer_distribution": retrieval_layers,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logging.info(f"Appended analysis record to {out_path}")


if __name__ == "__main__":
    main()
