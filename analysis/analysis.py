#!/usr/bin/env python3
"""
raptor_diagnostics.py

Consolidated, robust diagnostics for RAPTOR trees.

For each (dataset, seed), computes (skipping missing docs safely):
A) Tree structure (from *.meta.json when available)
B) Retrieval behavior using RAPTOR's collapsed-tree retrieval (robust to returning Node OR str):
   - layer selection distribution (when Node indices available)
   - leaf vs summary token fraction (best-effort)
   - raptor context tokens
   - (optional) baseline-vs-raptor overlap recall (requires cached leaf embeddings)
C) (optional) tree quality metrics from pickled Tree:
   - intra-cluster similarity (children within parent)
   - inter-cluster similarity (centroids per layer)
   - parent-child similarity
   - compression ratio

Aggregates mean/std across seeds PER DATASET.

Example:
  python analysis/raptor_diagnostics.py \
    --trees-root data/raptor_trees \
    --tree-subdir-template "seed{seed}_umap_raptor" \
    --baseline-embeds data/leaf_embeds \
    --do-overlap \
    --full-tree-metrics \
    --max-q 200 \
    --out results_raptor/diagnostics_umap_raptor.json
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional

import numpy as np
import tiktoken
import faiss

from raptor.tree_structures import Tree, Node
from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor.utils import split_text as raptor_split_text
from raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig

# -------------------------
# Defaults
# -------------------------

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
TOK = tiktoken.get_encoding("cl100k_base")

DATASET_SPLITS: Dict[str, str] = {
    "quality": "data/processed/quality/eval_val_sub50_q5.jsonl",
    "narrativeqa": "data/processed/narrativeqa/eval_val_sub50_q5.jsonl",
    "qasper": "data/processed/qasper/eval_val_sub50_q5.jsonl",
}

DEFAULT_DATASETS = ["quality", "narrativeqa", "qasper"]
DEFAULT_SEEDS = [224, 99, 42]

LEAF_CHUNK_TOKENS = 100
RETRIEVAL_BUDGET = 2000
RAPTOR_TOP_K = 50

_GLOBAL_SBERT = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")


# -------------------------
# Small helpers
# -------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _norm_text(x: Any) -> str:
    if isinstance(x, dict):
        for k in ("text", "question", "q", "context", "doc", "document", "doc_text"):
            if k in x and isinstance(x[k], str):
                return x[k]
        for v in x.values():
            if isinstance(v, str):
                return v
        return str(x)
    return str(x)

def doc_key_for_example(dataset: str, ex: Dict[str, Any], doc_text: str) -> str:
    """
    Match your earlier scripts:
    - QASPER: strip suffix after '.' to match tree filenames.
    """
    if dataset == "narrativeqa":
        return ex.get("document_id") or _sha1(doc_text)
    if dataset == "quality":
        return ex.get("article_id") or _sha1(doc_text)
    elif dataset == "qasper":
        return _sha1(doc_text)
    raise ValueError(f"Unknown dataset: {dataset}")

def _as_np(v: Any) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = _as_np(a); b = _as_np(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    return Xn @ Xn.T

def aggregate_stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None, "n": 0}
    arr = np.array(vals, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }

def mean_std_over_seeds(seed_means: List[Optional[float]]) -> Dict[str, Any]:
    clean = [float(x) for x in seed_means if x is not None and not np.isnan(x)]
    if not clean:
        return {"mean": None, "std": None, "n_seeds": 0}
    arr = np.array(clean, dtype=np.float32)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n_seeds": int(arr.size)}


# -------------------------
# Tree meta + quality metrics
# -------------------------

def read_meta_tree_stats(meta_json_path: Path) -> Optional[Dict[str, Any]]:
    if not meta_json_path.exists():
        return None
    try:
        meta = json.loads(meta_json_path.read_text(encoding="utf-8"))
        ts = meta.get("tree_stats", {}) or {}
        return {
            "layers_actual": ts.get("layers_actual"),
            "num_nodes": ts.get("num_nodes"),
            "num_leaves": ts.get("num_leaves"),
        }
    except Exception:
        return None

def analyze_tree_clusters_and_summaries(tree: Tree, embedding_key: str = "SBERT") -> Dict[str, Any]:
    intra_sims: List[float] = []
    inter_sims: List[float] = []
    parent_child_sims: List[float] = []
    compression_ratios: List[float] = []

    for layer, nodes in tree.layer_to_nodes.items():
        if layer == 0:
            continue

        centroids: List[np.ndarray] = []
        for parent in nodes:
            if not getattr(parent, "children", None):
                continue

            try:
                child_nodes: List[Node] = [tree.all_nodes[idx] for idx in parent.children]
            except Exception:
                continue

            if len(child_nodes) == 0:
                continue

            try:
                child_embs = np.stack([_as_np(ch.embeddings[embedding_key]) for ch in child_nodes], axis=0)
            except Exception:
                continue

            if child_embs.shape[0] >= 2:
                S = cosine_sim_matrix(child_embs)
                n = S.shape[0]
                vals = S[np.triu_indices(n, k=1)]
                if vals.size:
                    intra_sims.append(float(vals.mean()))

            try:
                parent_emb = _as_np(parent.embeddings[embedding_key])
                parent_child_sims.extend([cosine_sim(parent_emb, e) for e in child_embs])
            except Exception:
                pass

            child_text = " ".join((ch.text or "") for ch in child_nodes)
            child_tokens = len(TOK.encode(child_text))
            summary_tokens = len(TOK.encode(parent.text or ""))
            if child_tokens > 0:
                compression_ratios.append(summary_tokens / child_tokens)

            centroids.append(child_embs.mean(axis=0))

        if len(centroids) >= 2:
            C = np.stack(centroids, axis=0)
            S = cosine_sim_matrix(C)
            k = S.shape[0]
            vals = S[np.triu_indices(k, k=1)]
            if vals.size:
                inter_sims.append(float(vals.mean()))

    return {
        "intra_cluster_sim_docmean": aggregate_stats(intra_sims),
        "inter_cluster_sim_docmean": aggregate_stats(inter_sims),
        "parent_child_sim_docmean": aggregate_stats(parent_child_sims),
        "compression_ratio_docmean": aggregate_stats(compression_ratios),
    }


# -------------------------
# Overlap helpers
# -------------------------

def collect_leaf_indices(node_idx: int, tree: Tree, cache: Dict[int, Set[int]]) -> Set[int]:
    if node_idx in cache:
        return cache[node_idx]
    node = tree.all_nodes[node_idx]
    if not getattr(node, "children", None):
        res = {node_idx}
    else:
        res: Set[int] = set()
        for c in node.children:
            res |= collect_leaf_indices(c, tree, cache)
    cache[node_idx] = res
    return res

def baseline_retrieve_leaf_indices(
    leaf_embs: np.ndarray,
    leaf_texts: List[str],
    question: str,
    max_tokens: int = RETRIEVAL_BUDGET,
    top_k: int = 50,
) -> Tuple[int, List[int]]:
    if leaf_embs.size == 0:
        return 0, []
    index = faiss.IndexFlatIP(leaf_embs.shape[1])
    index.add(leaf_embs.astype(np.float32))
    q_vec = np.array([_GLOBAL_SBERT.create_embedding(question)], dtype=np.float32)
    k = min(len(leaf_texts), top_k)
    _, I = index.search(q_vec, k)

    total = 0
    picked: List[int] = []
    for raw in I[0]:
        idx = int(raw)
        t = leaf_texts[idx]
        nt = len(TOK.encode(t))
        if total + nt > max_tokens:
            break
        total += nt
        picked.append(idx)
    return total, picked


# -------------------------
# Build RA config once
# -------------------------

def build_ra_config_sbert() -> RetrievalAugmentationConfig:
    return RetrievalAugmentationConfig(
        tr_tokenizer=TOK,
        tr_threshold=0.5,
        tr_top_k=RAPTOR_TOP_K,
        tr_selection_mode="top_k",
        tr_context_embedding_model="SBERT",
        tr_embedding_model=_GLOBAL_SBERT,
        tr_num_layers=None,
        tr_start_layer=None,

        tb_tokenizer=TOK,
        tb_max_tokens=LEAF_CHUNK_TOKENS,
        tb_num_layers=1,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_summarization_model=None,
        tb_embedding_models={"SBERT": _GLOBAL_SBERT},
        tb_cluster_embedding_model="SBERT",
    )


# -------------------------
# Leaf-embed filename mapping
# -------------------------

def build_embed_map_for_seed(seed_dir: Path) -> Dict[str, Path]:
    """
    Maps base_id -> .npy path

    Handles cases like:
      1705.01234.npy  -> base_id "1705"
      1705.npy        -> base_id "1705"

    If multiple files share the same base_id, keeps the first (sorted) and logs a warning.
    """
    mp: Dict[str, Path] = {}
    if not seed_dir.exists():
        return mp
    paths = sorted(seed_dir.glob("*.npy"))
    collisions: Dict[str, List[Path]] = {}
    for p in paths:
        stem = p.stem  # "1705.01234" or "1705"
        base = stem.split(".")[0]
        if base in mp:
            collisions.setdefault(base, [mp[base]]).append(p)
            continue
        mp[base] = p

    if collisions:
        for base, ps in list(collisions.items())[:20]:
            logging.warning(f"[leaf_embeds] multiple .npy match base_id={base}: {[x.name for x in ps]}. Using {ps[0].name}")
    return mp

def load_leaf_embeds(seed_embed_map: Dict[str, Path], doc_key: str) -> Optional[np.ndarray]:
    p = seed_embed_map.get(doc_key)
    if p is None:
        return None
    try:
        return np.load(p)
    except Exception:
        return None


# -------------------------
# Per-seed runner
# -------------------------

def run_seed(
    dataset: str,
    split_path: Path,
    trees_root: Path,
    tree_subdir: str,
    seed: int,
    max_q: int,
    baseline_embeds_root: Optional[Path],
    do_overlap: bool,
    do_full_tree_metrics: bool,
    chunk_overlap: int,
) -> Dict[str, Any]:
    tree_dir = trees_root / dataset / tree_subdir
    if not tree_dir.exists():
        raise FileNotFoundError(f"Missing tree_dir: {tree_dir}")

    data = load_jsonl(split_path)

    layers_actual: List[float] = []
    num_nodes: List[float] = []
    num_leaves: List[float] = []
    avg_cluster_size: List[float] = []

    raptor_ctx_tokens: List[float] = []
    leaf_fracs: List[float] = []
    summary_fracs: List[float] = []
    overlap_recalls: List[float] = []
    baseline_ctx_tokens: List[float] = []
    layer_counts_total: Dict[int, int] = {}

    intra_docmeans: List[float] = []
    inter_docmeans: List[float] = []
    pc_docmeans: List[float] = []
    comp_docmeans: List[float] = []

    skips = {
        "missing_tree": 0,
        "bad_pickle": 0,
        "missing_meta": 0,
        "missing_embeddings": 0,
        "embeddings_len_mismatch": 0,
        "retrieval_failed": 0,
    }

    leaf_chunks_cache: Dict[str, List[str]] = {}
    leaf_embs_cache: Dict[str, Optional[np.ndarray]] = {}
    ra_cache: Dict[str, RetrievalAugmentation] = {}

    ra_cfg = build_ra_config_sbert()

    # Build embed map once per seed (fix for 1705.01234.npy vs 1705.pkl)
    seed_embed_map: Dict[str, Path] = {}
    if do_overlap:
        if baseline_embeds_root is None:
            raise ValueError("--do-overlap requires --baseline-embeds")
        seed_dir = baseline_embeds_root / f"seed{seed}"
        seed_embed_map = build_embed_map_for_seed(seed_dir)

    n_used = 0
    for ex in data:
        if n_used >= max_q:
            break

        doc_text = _norm_text(ex.get("doc_text", ""))
        question = _norm_text(ex.get("question", ""))
        if not doc_text or not question:
            continue

        doc_key = doc_key_for_example(dataset, ex, doc_text)

        pkl_path = tree_dir / f"{doc_key}.pkl"
        if not pkl_path.exists():
            skips["missing_tree"] += 1
            continue

        meta_path = tree_dir / f"{doc_key}.meta.json"
        ts = read_meta_tree_stats(meta_path)
        if ts is None:
            skips["missing_meta"] += 1
        else:
            la = ts.get("layers_actual")
            nn = ts.get("num_nodes")
            nl = ts.get("num_leaves")
            if la is not None: layers_actual.append(float(la))
            if nn is not None: num_nodes.append(float(nn))
            if nl is not None: num_leaves.append(float(nl))
            if nn is not None and nl is not None and nn > nl:
                internal = nn - nl
                if internal > 0:
                    avg_cluster_size.append(float(nl) / float(internal))

        if doc_key not in leaf_chunks_cache:
            leaf_chunks_cache[doc_key] = raptor_split_text(doc_text, TOK, LEAF_CHUNK_TOKENS, overlap=chunk_overlap)
        leaf_chunks = leaf_chunks_cache[doc_key]
        if not leaf_chunks:
            continue

        leaf_embs: Optional[np.ndarray] = None
        if do_overlap:
            if doc_key not in leaf_embs_cache:
                leaf_embs_cache[doc_key] = load_leaf_embeds(seed_embed_map, doc_key)
            leaf_embs = leaf_embs_cache[doc_key]
            if leaf_embs is None:
                skips["missing_embeddings"] += 1

        try:
            with pkl_path.open("rb") as f:
                tree: Tree = pickle.load(f)
        except Exception:
            skips["bad_pickle"] += 1
            continue

        if do_full_tree_metrics:
            try:
                tm = analyze_tree_clusters_and_summaries(tree, embedding_key="SBERT")
                if tm["intra_cluster_sim_docmean"]["mean"] is not None:
                    intra_docmeans.append(tm["intra_cluster_sim_docmean"]["mean"])
                if tm["inter_cluster_sim_docmean"]["mean"] is not None:
                    inter_docmeans.append(tm["inter_cluster_sim_docmean"]["mean"])
                if tm["parent_child_sim_docmean"]["mean"] is not None:
                    pc_docmeans.append(tm["parent_child_sim_docmean"]["mean"])
                if tm["compression_ratio_docmean"]["mean"] is not None:
                    comp_docmeans.append(tm["compression_ratio_docmean"]["mean"])
            except Exception:
                pass

        try:
            if doc_key not in ra_cache:
                ra_cache[doc_key] = RetrievalAugmentation(ra_cfg, tree=str(pkl_path))
            ra = ra_cache[doc_key]

            selected_nodes, _context = ra.retriever.retrieve_information_collapse_tree(
                question,
                top_k=RAPTOR_TOP_K,
                max_tokens=RETRIEVAL_BUDGET,
            )
        except Exception:
            skips["retrieval_failed"] += 1
            continue

        # Build node->layer mapping once per tree (best effort)
        node_to_layer: Dict[int, int] = {}
        try:
            for L, nodes in tree.layer_to_nodes.items():
                for n in nodes:
                    idx = getattr(n, "index", None)
                    if isinstance(idx, int):
                        node_to_layer[idx] = int(L)
        except Exception:
            pass

        leaf_node_indices = set(tree.leaf_nodes.keys())
        leaf_tokens = 0
        summary_tokens = 0
        leaf_cache: Dict[int, Set[int]] = {}
        raptor_leaf_set: Set[int] = set()

        for item in selected_nodes:
            # Robust to Node OR str
            if isinstance(item, str):
                txt = item
                idx = None
            else:
                txt = getattr(item, "text", "") or ""
                idx = getattr(item, "index", None)

            nt = len(TOK.encode(txt))

            if isinstance(idx, int) and idx in leaf_node_indices:
                leaf_tokens += nt
            else:
                # If we don't have a valid node index, treat as summary/unknown
                summary_tokens += nt

            if isinstance(idx, int):
                # layer dist
                if idx in node_to_layer:
                    L = node_to_layer[idx]
                    layer_counts_total[L] = layer_counts_total.get(L, 0) + 1
                # leaf expansion
                try:
                    raptor_leaf_set |= collect_leaf_indices(idx, tree, leaf_cache)
                except Exception:
                    pass

        total_tokens = leaf_tokens + summary_tokens
        raptor_ctx_tokens.append(float(total_tokens))
        leaf_fracs.append(float(leaf_tokens / total_tokens) if total_tokens > 0 else 0.0)
        summary_fracs.append(float(summary_tokens / total_tokens) if total_tokens > 0 else 0.0)

        if do_overlap and leaf_embs is not None:
            if leaf_embs.shape[0] != len(leaf_chunks):
                skips["embeddings_len_mismatch"] += 1
                logging.warning(
                    f"[{dataset} seed{seed}] EMBED LEN MISMATCH doc_key={doc_key} "
                    f"embeds={leaf_embs.shape[0]} chunks={len(leaf_chunks)} "
                    f"(chunk_tokens={LEAF_CHUNK_TOKENS}, overlap={chunk_overlap})"
                )
            else:
                base_tok, base_idx = baseline_retrieve_leaf_indices(
                    leaf_embs, leaf_chunks, question, max_tokens=RETRIEVAL_BUDGET, top_k=50
                )
                baseline_set = set(base_idx)
                if baseline_set:
                    overlap = len(baseline_set & raptor_leaf_set) / len(baseline_set)
                    overlap_recalls.append(float(overlap))
                    baseline_ctx_tokens.append(float(base_tok))

        n_used += 1

    total_sel = sum(layer_counts_total.values()) or 1
    layer_dist = {
        str(L): {"count": int(c), "fraction": float(c) / float(total_sel)}
        for L, c in sorted(layer_counts_total.items(), key=lambda x: x[0])
    }

    return {
        "dataset": dataset,
        "seed": seed,
        "tree_dir": str(tree_dir),
        "n_questions_used": n_used,
        "skips": skips,

        "tree_stats": {
            "layers_actual": aggregate_stats(layers_actual),
            "num_nodes": aggregate_stats(num_nodes),
            "num_leaves": aggregate_stats(num_leaves),
            "avg_cluster_size": aggregate_stats(avg_cluster_size),
        },

        "retrieval_stats": {
            "raptor_ctx_tokens": aggregate_stats(raptor_ctx_tokens),
            "leaf_token_frac": aggregate_stats(leaf_fracs),
            "summary_token_frac": aggregate_stats(summary_fracs),
            "layer_distribution": layer_dist,
            "overlap_recall": aggregate_stats(overlap_recalls) if do_overlap else None,
            "baseline_ctx_tokens": aggregate_stats(baseline_ctx_tokens) if do_overlap else None,
        },

        "tree_quality": {
            "intra_cluster_sim_docmean": aggregate_stats(intra_docmeans) if do_full_tree_metrics else None,
            "inter_cluster_sim_docmean": aggregate_stats(inter_docmeans) if do_full_tree_metrics else None,
            "parent_child_sim_docmean": aggregate_stats(pc_docmeans) if do_full_tree_metrics else None,
            "compression_ratio_docmean": aggregate_stats(comp_docmeans) if do_full_tree_metrics else None,
        },
    }


# -------------------------
# Aggregate per dataset across seeds
# -------------------------

def aggregate_dataset_over_seeds(per_seed: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    def seed_means(path: List[str]) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for _, res in per_seed.items():
            cur: Any = res
            ok = True
            for p in path:
                if cur is None or p not in cur:
                    ok = False
                    break
                cur = cur[p]
            if not ok or cur is None:
                continue
            m = cur.get("mean", None)
            out.append(m if isinstance(m, (int, float)) else None)
        return out

    agg: Dict[str, Any] = {}

    for k in ["layers_actual", "num_nodes", "num_leaves", "avg_cluster_size"]:
        agg[f"tree_stats.{k}.mean_over_seeds"] = mean_std_over_seeds(seed_means(["tree_stats", k]))

    for k in ["raptor_ctx_tokens", "leaf_token_frac", "summary_token_frac"]:
        agg[f"retrieval_stats.{k}.mean_over_seeds"] = mean_std_over_seeds(seed_means(["retrieval_stats", k]))

    ov = seed_means(["retrieval_stats", "overlap_recall"])
    agg["retrieval_stats.overlap_recall.mean_over_seeds"] = mean_std_over_seeds(ov) if any(v is not None for v in ov) else None

    for k in [
        "intra_cluster_sim_docmean",
        "inter_cluster_sim_docmean",
        "parent_child_sim_docmean",
        "compression_ratio_docmean",
    ]:
        vals = seed_means(["tree_quality", k])
        agg[f"tree_quality.{k}.mean_over_seeds"] = mean_std_over_seeds(vals) if any(v is not None for v in vals) else None

    layer_counts_total: Dict[int, int] = {}
    total_sel = 0
    for _, res in per_seed.items():
        ld = res["retrieval_stats"]["layer_distribution"]
        for layer_str, info in ld.items():
            L = int(layer_str)
            c = int(info.get("count", 0))
            layer_counts_total[L] = layer_counts_total.get(L, 0) + c
            total_sel += c
    total_sel = total_sel or 1
    agg["retrieval_stats.layer_distribution_agg"] = {
        str(L): {"count": int(c), "fraction": float(c) / float(total_sel)}
        for L, c in sorted(layer_counts_total.items(), key=lambda x: x[0])
    }

    skip_totals: Dict[str, int] = {}
    for _, res in per_seed.items():
        for k, v in res.get("skips", {}).items():
            skip_totals[k] = skip_totals.get(k, 0) + int(v)
    agg["skips_total_over_seeds"] = skip_totals

    return agg


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees-root", default="data/raptor_trees", type=str)
    ap.add_argument("--tree-subdir-template", required=True, type=str,
                    help='e.g. "seed{seed}_umap_raptor" or "seed{seed}_none_hdbscan"')
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=list(DATASET_SPLITS.keys()))
    ap.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS, type=int)
    ap.add_argument("--max-q", default=200, type=int)
    ap.add_argument("--chunk-overlap", default=0, type=int,
                    help="Must match eval chunking overlap if you use --do-overlap.")
    ap.add_argument("--baseline-embeds", default=None, type=str,
                    help="Root like data/leaf_embeds. Required for --do-overlap.")
    ap.add_argument("--do-overlap", action="store_true")
    ap.add_argument("--full-tree-metrics", action="store_true")
    ap.add_argument("--out", required=True, type=str)

    args = ap.parse_args()

    trees_root = Path(args.trees_root)
    baseline_root = Path(args.baseline_embeds) if args.baseline_embeds else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_out: Dict[str, Any] = {
        "trees_root": str(trees_root),
        "tree_subdir_template": args.tree_subdir_template,
        "datasets": {},
        "seeds": args.seeds,
        "max_q": args.max_q,
        "chunk_overlap": args.chunk_overlap,
        "do_overlap": args.do_overlap,
        "full_tree_metrics": args.full_tree_metrics,
    }

    for dataset in args.datasets:
        split_path = Path(DATASET_SPLITS[dataset])
        if not split_path.exists():
            logging.warning(f"[{dataset}] Missing split: {split_path}, skipping dataset.")
            continue

        per_seed: Dict[int, Dict[str, Any]] = {}
        for seed in args.seeds:
            subdir = args.tree_subdir_template.format(seed=seed, dataset=dataset)
            logging.info(f"[{dataset}] seed={seed} subdir={subdir}")
            try:
                res = run_seed(
                    dataset=dataset,
                    split_path=split_path,
                    trees_root=trees_root,
                    tree_subdir=subdir,
                    seed=seed,
                    max_q=args.max_q,
                    baseline_embeds_root=baseline_root,
                    do_overlap=args.do_overlap,
                    do_full_tree_metrics=args.full_tree_metrics,
                    chunk_overlap=args.chunk_overlap,
                )
                per_seed[seed] = res
            except FileNotFoundError as e:
                logging.warning(f"[{dataset}] seed={seed} missing dir, skipping seed: {e}")
            except Exception as e:
                logging.warning(f"[{dataset}] seed={seed} error, skipping seed: {e}")

        if not per_seed:
            logging.warning(f"[{dataset}] No seeds collected, skipping dataset aggregation.")
            continue

        all_out["datasets"][dataset] = {
            "per_seed": per_seed,
            "aggregate_over_seeds": aggregate_dataset_over_seeds(per_seed),
        }

    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    logging.info(f"Wrote diagnostics to {out_path}")

if __name__ == "__main__":
    main()
