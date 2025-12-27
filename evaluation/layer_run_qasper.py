#!/usr/bin/env python3
"""
QASPER RAPTOR depth ablation (hierarchy depth).
"""

import os, json, argparse, re, string, hashlib, logging, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# -------------------------
# Determinism
# -------------------------
def set_global_seed(seed: int = 224):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# Fixed seeds for this ablation (match your QuALITY ablation)
ABLATION_SEEDS = [224, 42, 99]

# -------------------------
# Constants (match your eval)
# -------------------------
LEAF_CHUNK_TOKENS = 100
RETRIEVAL_BUDGET = 2000
UQA_MODEL_NAME = "allenai/unifiedqa-v2-t5-3b-1363200"
UQA_MAX_LEN = 512
UQA_CONTEXT_BUDGET = 400
UQA_SAFETY = 12
RAPTOR_TOP_K = 50

SUMMARY_MAX_TOKENS = 100
TREE_TOP_K = 5
TREE_THRESHOLD = 0.5
REDUCTION_DIM = 10

# -------------------------
# RAPTOR + Models
# -------------------------
from raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.cluster_tree_builder import ClusterTreeConfig
from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor.SummarizationModels import GPT3TurboSummarizationModel
from raptor.QAModels import UnifiedQAModel
from raptor.cluster_utils import RAPTOR_Clustering, HDBSCAN_Clustering

# Tokenization / metrics
import nltk
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab/english/")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

import tiktoken
TOK = tiktoken.get_encoding("cl100k_base")

from transformers import AutoTokenizer
_UQA_TOK = AutoTokenizer.from_pretrained(UQA_MODEL_NAME, use_fast=False)
if not getattr(_UQA_TOK, "model_max_length", None):
    _UQA_TOK.model_max_length = UQA_MAX_LEN

# Repo chunker (tree builder uses this internally, baseline not needed here)
from raptor.utils import split_text

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# -------------------------
# IO helpers
# -------------------------
def load_jsonl(path_jsonl: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def append_jsonl(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

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
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _tok_len_cl100k(s: str) -> int:
    return len(TOK.encode(s or ""))

# -------------------------
# UnifiedQA clipping (identical to your eval)
# -------------------------
def clip_for_unifiedqa(question: str, context: str, budget: int | None = None) -> Tuple[str, str]:
    tok = _UQA_TOK
    max_len = budget or int(tok.model_max_length or UQA_MAX_LEN)
    target = max(32, max_len - UQA_SAFETY)
    sep = " \n "

    q_ids = tok.encode(str(question).strip(), add_special_tokens=False)
    c_ids = tok.encode(str(context).strip(), add_special_tokens=False)
    sep_ids = tok.encode(sep, add_special_tokens=False)

    # keep context near requested budget first
    c_ids = c_ids[:UQA_CONTEXT_BUDGET]

    def total_len(qi, ci):
        return len(qi) + len(sep_ids) + len(ci)

    qi, ci = q_ids, c_ids

    # Trim if needed to satisfy total <= ~max_len
    if total_len(qi, ci) > target:
        over = total_len(qi, ci) - target
        if over > 0:
            ci = ci[: max(0, len(ci) - over)]
    if total_len(qi, ci) > target:
        over = total_len(qi, ci) - target
        qi = qi[: max(0, len(qi) - over)]

    q_trim = tok.decode(qi, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    c_trim = tok.decode(ci, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return q_trim, c_trim

# -------------------------
# SQuAD-style F1 for QASPER (identical to your eval)
# -------------------------
_ARTS = {"a", "an", "the"}
_PUNC_TABLE = str.maketrans("", "", string.punctuation)

def _normalize(s: str) -> List[str]:
    s = s.lower()
    s = s.translate(_PUNC_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split()
    return [t for t in toks if t not in _ARTS]

def f1_answer(pred: str, golds: List[str]) -> float:
    def f1(a, b):
        A, B = _normalize(a), _normalize(b)
        if not A or not B:
            return 0.0
        common = 0
        for w in set(A + B):
            common += min(A.count(w), B.count(w))
        if common == 0:
            return 0.0
        prec = common / len(A)
        rec = common / len(B)
        return 2 * prec * rec / (prec + rec)

    return max((f1(pred, g) for g in golds), default=0.0)

# -------------------------
# RAPTOR config builder (depth-controlled)
# -------------------------
def build_ra_with_sbert_config(
    *,
    num_layers: int,
    dr_method: str = "umap",
    cluster_algo: str = "raptor",
) -> RetrievalAugmentationConfig:
    """
    Mirrors your earlier build_ra_with_sbert_config but adds num_layers control.
    """
    sbert = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")

    cluster_algo = cluster_algo.lower()
    if cluster_algo == "raptor":
        clustering_algorithm = RAPTOR_Clustering
        clustering_params = {"dr_method": dr_method}
    elif cluster_algo == "hdbscan":
        clustering_algorithm = HDBSCAN_Clustering
        clustering_params = {
            "dr_method": dr_method,
            "min_cluster_size": 5,
            "min_samples": 1,
            "metric": "euclidean",
        }
    else:
        raise ValueError(f"Unknown cluster_algo: {cluster_algo}")

    tb_cfg = ClusterTreeConfig(
        tokenizer=TOK,
        max_tokens=LEAF_CHUNK_TOKENS,
        num_layers=int(num_layers),
        threshold=TREE_THRESHOLD,
        top_k=TREE_TOP_K,
        selection_mode="top_k",
        summarization_length=SUMMARY_MAX_TOKENS,
        summarization_model=GPT3TurboSummarizationModel("gpt-3.5-turbo"),
        embedding_models={"SBERT": sbert},
        cluster_embedding_model="SBERT",
        reduction_dimension=REDUCTION_DIM,
        clustering_algorithm=clustering_algorithm,
        clustering_params=clustering_params,
    )

    return RetrievalAugmentationConfig(
        tree_builder_config=tb_cfg,
        tr_tokenizer=TOK,
        tr_threshold=TREE_THRESHOLD,
        tr_top_k=RAPTOR_TOP_K,
        tr_selection_mode="top_k",
        tr_context_embedding_model="SBERT",
        tr_embedding_model=sbert,
        tb_tokenizer=TOK,
    )

# -------------------------
# Tree cache (depth-aware dir)
# -------------------------
class RAPTORCache:
    """One built tree per document per seed. Stores <doc_key>.pkl and <doc_key>.meta.json."""
    def __init__(self, cfg: RetrievalAugmentationConfig, cache_dir: Path, seed: int, num_layers: int):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.num_layers = int(num_layers)

    def _paths(self, doc_key: str):
        base = self.cache_dir / f"{doc_key}"
        return base.with_suffix(".pkl"), base.with_suffix(".meta.json")

    def get_or_build(self, doc_text: str, doc_key: str) -> RetrievalAugmentation:
        pkl, meta = self._paths(doc_key)
        if pkl.exists():
            return RetrievalAugmentation(self.cfg, tree=str(pkl))

        ra = RetrievalAugmentation(self.cfg)
        ra.add_documents(doc_text)
        ra.save(str(pkl))

        tb = self.cfg.tree_builder_config
        tr = self.cfg.tree_retriever_config
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({
                "seed": self.seed,
                "dataset": "qasper",
                "leaf_chunk_tokens": tb.max_tokens,
                "num_layers_cap": tb.num_layers,
                "tree_threshold": tb.threshold,
                "tree_top_k": tb.top_k,
                "selection_mode": tb.selection_mode,
                "summarizer": "gpt-3.5-turbo",
                "summary_len_param": tb.summarization_length,
                "embed_model": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                "clustering_params": getattr(tb, "clustering_params", None),
                "reduction_dimension": getattr(tb, "reduction_dimension", None),
                "retrieval": {
                    "mode": "collapsed_tree",
                    "tr_top_k": tr.top_k,
                    "budget_tokens": RETRIEVAL_BUDGET,
                    "context_embed_model": tr.context_embedding_model,
                },
                "uqa_context_tokens": UQA_CONTEXT_BUDGET,
                "tree_stats": {
                    "layers_actual": getattr(ra.tree, "num_layers", None),
                    "num_nodes": len(getattr(ra.tree, "all_nodes", {})),
                    "num_leaves": len(getattr(ra.tree, "leaf_nodes", {})),
                }
            }, f, ensure_ascii=False, indent=2)

        return RetrievalAugmentation(self.cfg, tree=str(pkl))

# -------------------------
# Debug helper (same behavior)
# -------------------------
def _maybe_debug_dump(example_idx: int, q: str, ctx: str, pred: str, refs: List[str]):
    if os.environ.get("DEBUG_EVAL", "") != "1":
        return
    print("\n--- DEBUG SAMPLE ---")
    print(f"idx={example_idx}")
    print("Q:", q[:400])
    print("CTX:", ctx[:400].replace("\n", " ") + ("..." if len(ctx) > 400 else ""))
    print("PRED:", repr(pred))
    print("REFS:", refs[:4])
    print("--- END DEBUG ---\n")

# -------------------------
# Runner: QASPER RAPTOR only
# -------------------------
def run_qasper_raptor_only(
    path: Path,
    tree_dir: Path,
    seed: int,
    num_layers: int,
    dr_method: str,
    cluster_algo: str,
) -> Dict:
    data = load_jsonl(path)
    qa = UnifiedQAModel(UQA_MODEL_NAME)

    cfg = build_ra_with_sbert_config(
        num_layers=num_layers,
        dr_method=dr_method,
        cluster_algo=cluster_algo,
    )
    cache = RAPTORCache(cfg, tree_dir, seed=seed, num_layers=num_layers)

    f1s: List[float] = []
    empties = 0
    n_used = 0

    for i, ex in enumerate(tqdm(data, desc=f"QASPER seed={seed} L={num_layers}")):
        doc = _norm_text(ex["doc_text"])
        q = _norm_text(ex["question"])
        refs = [_norm_text(r) for r in (ex.get("gold_answers", []) or [])]

        # Match your normal script's doc_key logic (paper_id or sha1)
        doc_key = ex.get("paper_id") or _sha1(doc)

        ra = cache.get_or_build(doc, doc_key)
        ctx = ra.retrieve(
            q,
            top_k=RAPTOR_TOP_K,
            max_tokens=RETRIEVAL_BUDGET,
            collapse_tree=True,
            return_layer_information=False,
        )

        q_trim, c_trim = clip_for_unifiedqa(q, _norm_text(ctx), budget=UQA_MAX_LEN)
        if os.environ.get("DEBUG_EVAL") == "1":
            logging.info(f"[dbg] raw ctx tokens: {_tok_len_cl100k(_norm_text(ctx))}")
            logging.info(f"[dbg] clipped ctx tokens: {_tok_len_cl100k(c_trim)}")
            logging.info(f"[dbg] ctx sha1: {_sha1(c_trim)}")

        pred = qa.answer_question(c_trim, q_trim)
        if not pred.strip():
            empties += 1

        _maybe_debug_dump(i, q_trim, c_trim, pred, refs)

        if refs:
            f1s.append(f1_answer(pred, refs))
            n_used += 1

    return {
        "f1": float(np.mean(f1s) * 100 if f1s else 0.0),
        "empty_preds": int(empties),
        "n": int(n_used),  # keep comparable to your qasper metric loop
    }

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="QASPER eval JSONL (e.g., data/processed/qasper/eval_val.jsonl)")
    ap.add_argument("--num-layers", type=int, required=True, help="Hierarchy depth (e.g., 1, 3, 8)")
    ap.add_argument("--out", default="results/table_runs.jsonl", help="Append JSONL results here.")
    ap.add_argument("--tree-dir", default="data/raptor_trees", help="Base dir to cache/load RAPTOR trees")
    ap.add_argument("--run-name", default="", help="Optional tag stored with results.")
    ap.add_argument("--dr-method", choices=["umap", "pacmap", "trimap", "none"], default="umap")
    ap.add_argument("--clusterer", choices=["raptor", "hdbscan"], default="raptor")
    args = ap.parse_args()

    path = Path(args.split)
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path.resolve()}")

    for seed in ABLATION_SEEDS:
        set_global_seed(seed)

        # Depth-aware cache subdir so different L don't mix or overwrite existing trees.
        seed_tag = f"seed{seed}"
        tree_subdir = f"{seed_tag}_{args.dr_method}_{args.clusterer}_L{int(args.num_layers)}"
        tree_dir = Path(args.tree_dir) / "qasper" / tree_subdir

        t0 = time.time()
        metrics = run_qasper_raptor_only(
            path=path,
            tree_dir=tree_dir,
            seed=seed,
            num_layers=int(args.num_layers),
            dr_method=args.dr_method,
            cluster_algo=args.clusterer,
        )
        elapsed = time.time() - t0

        record = {
            "timestamp": int(time.time()),
            "run_name": args.run_name or None,
            "dataset": "qasper",
            "split": str(path),
            "with_raptor": True,
            "seed": int(seed),
            "retrieval_budget": RETRIEVAL_BUDGET,
            "uqa_context_budget": UQA_CONTEXT_BUDGET,
            "tb_max_tokens": LEAF_CHUNK_TOKENS,
            "num_layers": int(args.num_layers),
            "dr_method": args.dr_method,
            "clusterer": args.clusterer,
            "tree_cache_subdir": str(tree_subdir),
            "metrics": metrics,
            "elapsed_sec": round(elapsed, 2),
        }

        print(json.dumps(record, indent=2))
        append_jsonl(Path(args.out), record)

if __name__ == "__main__":
    main()
