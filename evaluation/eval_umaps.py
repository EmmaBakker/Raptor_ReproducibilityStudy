#!/usr/bin/env python3

import os, json, argparse, re, string, hashlib, logging, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

# Determinism
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

DEFAULT_SEED = 224

# Constants
LEAF_CHUNK_TOKENS = 100
RETRIEVAL_BUDGET = 2000
UQA_MODEL_NAME = "allenai/unifiedqa-v2-t5-3b-1363200"
UQA_MAX_LEN = 512
UQA_CONTEXT_BUDGET = 400
UQA_SAFETY = 12
RAPTOR_TOP_K = 50 

# RAPTOR + Models
from raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.cluster_tree_builder import ClusterTreeConfig
from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor.SummarizationModels import GPT3TurboSummarizationModel
from raptor.QAModels import UnifiedQAModel
from raptor.cluster_utils import RAPTOR_Clustering, HDBSCAN_Clustering

# Tokenization / metrics
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

import tiktoken
TOK = tiktoken.get_encoding("cl100k_base")
from transformers import AutoTokenizer
_UQA_TOK = AutoTokenizer.from_pretrained(UQA_MODEL_NAME, use_fast=False)
if not getattr(_UQA_TOK, "model_max_length", None):
    _UQA_TOK.model_max_length = UQA_MAX_LEN

# Import the repo’s own chunker (faithful to their baseline + builder)
from raptor.utils import split_text

# IO helpers
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
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# UnifiedQA clipping utils
def trim_with_uqa_tok(s: str, max_tokens: int) -> str:
    ids = _UQA_TOK.encode(s, add_special_tokens=False)
    ids = ids[:max_tokens]
    return _UQA_TOK.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def _tok_len_cl100k(s: str) -> int:
    return len(TOK.encode(s or ""))

def clip_for_unifiedqa(question: str, context: str, budget: int | None = None) -> Tuple[str, str]:
    tok = _UQA_TOK
    max_len = budget or int(tok.model_max_length or UQA_MAX_LEN)
    target = max(32, max_len - UQA_SAFETY)
    sep = " \n "

    q_ids = tok.encode(str(question).strip(), add_special_tokens=False)
    c_ids = tok.encode(str(context).strip(), add_special_tokens=False)
    sep_ids = tok.encode(sep, add_special_tokens=False)

    # Keep context near requested budget first
    c_ids = c_ids[:UQA_CONTEXT_BUDGET]

    def total_len(qi, ci): return len(qi) + len(sep_ids) + len(ci)
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

# Metrics
def bleu1(refs: List[str], hyp: str) -> float:
    ref_tok = [word_tokenize(r) for r in refs]
    hyp_tok = word_tokenize(hyp)
    ch = SmoothingFunction().method3
    return sentence_bleu(ref_tok, hyp_tok, weights=(1.0, 0, 0, 0), smoothing_function=ch) * 100

def bleu4_equal(refs: List[str], hyp: str) -> float:
    ref_tok = [word_tokenize(r) for r in refs]
    hyp_tok = word_tokenize(hyp)
    ch = SmoothingFunction().method3
    return sentence_bleu(ref_tok, hyp_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=ch) * 100

def rougeL(refs: List[str], hyp: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(r, hyp)['rougeL'].fmeasure for r in refs]
    return (max(scores) * 100) if scores else 0.0

def meteor_tokenized(refs: List[str], hyp: str) -> float:
    ref_tokens = [word_tokenize(r) for r in refs]
    hyp_tokens = word_tokenize(hyp)
    return meteor_score(ref_tokens, hyp_tokens) * 100

# SQuAD-style F1 for QASPER
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
        if not A or not B: return 0.0
        common = 0
        for w in set(A + B):
            common += min(A.count(w), B.count(w))
        if common == 0: return 0.0
        prec = common / len(A)
        rec  = common / len(B)
        return 2 * prec * rec / (prec + rec)
    return max((f1(pred, g) for g in golds), default=0.0)

# RAPTOR config & caching
def build_ra_with_sbert_config(
    dr_method: str = "umap",
    cluster_algo: str = "raptor",
) -> RetrievalAugmentationConfig:
    """
    Build RA config that mirrors the repo (SBERT embeddings, GPT-3.5 summaries,
    cluster builder) but lets you choose:
      - dr_method: 'umap', 'pacmap', 'trimap', or 'none'
      - cluster_algo: 'raptor' (original GMM) or 'hdbscan'
    """
    sbert = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")

    # Choose clustering algorithm and its params
    cluster_algo = cluster_algo.lower()
    if cluster_algo == "raptor":
        # Original RAPTOR two-stage GMM clustering
        clustering_algorithm = RAPTOR_Clustering
        clustering_params = {
            # only pass DR method; threshold comes from ClusterTreeConfig.threshold
            "dr_method": dr_method,   # "umap", "pacmap", "trimap"
            }

    elif cluster_algo == "hdbscan":
        clustering_algorithm = HDBSCAN_Clustering
        clustering_params = {
            "dr_method": dr_method,   # "umap" or "none" for your experiments
            "min_cluster_size": 5,
            "min_samples": 1,
            "metric": "euclidean",    # always Euclidean for HDBSCAN
        }

    else:
        raise ValueError(f"Unknown cluster_algo: {cluster_algo}")

    # Explicitly use ClusterTreeConfig so we record reduction_dimension, etc.
    tb_cfg = ClusterTreeConfig(
        tokenizer=TOK,
        max_tokens=LEAF_CHUNK_TOKENS,
        num_layers=8,
        threshold=0.5,          # GMM membership prob threshold (as in your logs)
        top_k=5,
        selection_mode="top_k",
        summarization_length=100,
        summarization_model=GPT3TurboSummarizationModel("gpt-3.5-turbo"),
        embedding_models={"SBERT": sbert},
        cluster_embedding_model="SBERT",
        reduction_dimension=10,
        clustering_algorithm=clustering_algorithm,
        clustering_params=clustering_params,
    )

    return RetrievalAugmentationConfig(
        tree_builder_config=tb_cfg,
        tr_tokenizer=TOK,
        tr_threshold=0.5,
        tr_top_k=RAPTOR_TOP_K,
        tr_selection_mode="top_k",
        tr_context_embedding_model="SBERT",
        tr_embedding_model=sbert,
        tb_tokenizer=TOK,
    )


class RAPTORCache:
    """One built tree per document per seed. Stores <doc_key>.pkl and <doc_key>.meta.json."""
    def __init__(self, cfg: RetrievalAugmentationConfig, cache_dir: Path, seed: int):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

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
        # richer meta for audit
        tb = self.cfg.tree_builder_config
        tr = self.cfg.tree_retriever_config
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({
                "seed": self.seed,
                "leaf_chunk_tokens": tb.max_tokens,
                "num_layers_cap": tb.num_layers,
                "reduction_dimension": getattr(tb, "reduction_dimension", None),
                "clustering_params": getattr(tb, "clustering_params", None),
                "tree_threshold": tb.threshold,
                "tree_top_k": tb.top_k,
                "selection_mode": tb.selection_mode,
                "summarizer": "gpt-3.5-turbo",
                "summary_len_param": tb.summarization_length,
                "embed_model": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                "retrieval": {
                    "mode": "collapsed_tree",
                    "tr_top_k": tr.top_k,
                    "budget_tokens": RETRIEVAL_BUDGET,
                    "context_embed_model": tr.context_embedding_model
                },
                "uqa_context_tokens": UQA_CONTEXT_BUDGET,
                "tree_stats": {
                    "layers_actual": getattr(ra.tree, "num_layers", None),
                    "num_nodes": len(getattr(ra.tree, "all_nodes", {})),
                    "num_leaves": len(getattr(ra.tree, "leaf_nodes", {})),
                }
            }, f, ensure_ascii=False, indent=2)
        return RetrievalAugmentation(self.cfg, tree=str(pkl))

# SBERT baseline caching (leaf embeddings) – repo-faithful (NO L2 normalization)
def _leaf_embed_cache_path(root: Path, doc_key: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{doc_key}.npy"

def baseline_sbert_context_cached(
    leaf_chunks: List[str],
    question: str,
    embed_model: SBertEmbeddingModel,
    tokenizer,
    cache_root: Path,
    doc_key: str,
    max_tokens=RETRIEVAL_BUDGET
) -> str:
    import faiss
    if not leaf_chunks:
        return ""
    path = _leaf_embed_cache_path(cache_root, doc_key)
    if path.exists():
        embs = np.load(path)  # (N, D), raw (no L2 norm) to mirror FaissRetriever
    else:
        embs = np.array([embed_model.create_embedding(_norm_text(t)) for t in leaf_chunks],
                        dtype=np.float32)
        np.save(path, embs)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    q_vec = np.array([embed_model.create_embedding(_norm_text(question))], dtype=np.float32)
    # no normalization → matches repo FaissRetriever

    k = min(len(leaf_chunks), 50)
    _, I = index.search(q_vec, k)

    total_tokens, picked = 0, []
    for i in I[0]:
        piece = _norm_text(leaf_chunks[int(i)])
        piece_tokens = len(tokenizer.encode(piece))
        if total_tokens + piece_tokens > max_tokens:
            break
        picked.append(piece)
        total_tokens += piece_tokens

    return "\n\n".join(picked)

# Debug helper
def _maybe_debug_dump(example_idx: int, q: str, ctx: str, pred: str, refs_or_choices):
    if os.environ.get("DEBUG_EVAL", "") != "1":
        return
    print("\n--- DEBUG SAMPLE ---")
    print(f"idx={example_idx}")
    print("Q:", q[:400])
    print("CTX:", ctx[:400].replace("\n", " ") + ("..." if len(ctx) > 400 else ""))
    print("PRED:", repr(pred))
    if isinstance(refs_or_choices, list):
        print("REFS/CHOICES:", refs_or_choices[:4])
    print("--- END DEBUG ---\n")

# Runners
def run_narrativeqa(path: Path, with_raptor: bool, tree_dir: Path, embed_cache_dir: Path, seed: int, dr_method: str, cluster_algo: str) -> Dict:
    data = load_jsonl(path)
    qa   = UnifiedQAModel(UQA_MODEL_NAME)
    sbert = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")
    cfg = build_ra_with_sbert_config(dr_method=dr_method, cluster_algo=cluster_algo)
    cache = RAPTORCache(cfg, tree_dir, seed)

    scores = {"bleu1": [], "bleu4": [], "rougeL": [], "meteor": []}
    empties = 0

    for i, ex in enumerate(tqdm(data, desc="NarrativeQA")):
        doc  = _norm_text(ex["doc_text"])
        q    = _norm_text(ex["question"])
        refs = [_norm_text(r) for r in (ex.get("gold_answers", []) or [])]
        doc_key = ex.get("document_id") or _sha1(doc)

        if with_raptor:
            ra = cache.get_or_build(doc, doc_key)
            ctx = ra.retrieve(q, top_k=RAPTOR_TOP_K, max_tokens=RETRIEVAL_BUDGET,
                              collapse_tree=True, return_layer_information=False)
        else:
            chunks = split_text(doc, TOK, LEAF_CHUNK_TOKENS)
            ctx = baseline_sbert_context_cached(
                chunks, q, sbert, TOK, embed_cache_dir, doc_key, RETRIEVAL_BUDGET
            )

        # paper: ~400-token context to UQA (clip with UQA tokenizer)
        q_trim, c_trim = clip_for_unifiedqa(q, _norm_text(ctx), budget=UQA_MAX_LEN)
        pred = qa.answer_question(c_trim, q_trim)

        if not pred.strip(): empties += 1
        _maybe_debug_dump(i, q_trim, c_trim, pred, refs)

        if refs:
            scores["bleu1"].append(bleu1(refs, pred))
            scores["bleu4"].append(bleu4_equal(refs, pred))
            scores["rougeL"].append(rougeL(refs, pred))
            scores["meteor"].append(meteor_tokenized(refs, pred))

    return {
        "bleu1": float(np.mean(scores["bleu1"]) if scores["bleu1"] else 0.0),
        "bleu4": float(np.mean(scores["bleu4"]) if scores["bleu4"] else 0.0),
        "rougeL": float(np.mean(scores["rougeL"]) if scores["rougeL"] else 0.0),
        "meteor": float(np.mean(scores["meteor"]) if scores["meteor"] else 0.0),
        "empty_preds": int(empties),
        "n": int(len(data)),
    }

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def _quality_prompt(question: str, choices: List[str]) -> str:
    lettered = "\n".join([f"({chr(ord('A')+i)}) {c}" for i, c in enumerate(choices)])
    return f"{question}\n\nOptions:\n{lettered}\n\nAnswer with the option letter."

def _parse_quality_pred(pred: str, choices: List[str]) -> int:
    p = pred.strip().upper()
    for i, L in enumerate(_LETTERS[:len(choices)]):
        if re.search(rf"\b{L}\b", p):
            return i
    m = re.search(r"\bOPTION\s+([A-Z])\b", p)
    if m:
        i = ord(m.group(1)) - ord('A')
        if 0 <= i < len(choices): return i
    m = re.search(r"\b([0-9])\b", p)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(choices): return idx
    # fallback: overlap
    p_low = pred.lower()
    sims = []
    for i, c in enumerate(choices):
        ctoks = set(word_tokenize(c.lower()))
        ptoks = set(word_tokenize(p_low))
        sims.append(len(ctoks & ptoks))
    return int(np.argmax(sims)) if sims else 0

def run_quality(path: Path, with_raptor: bool, tree_dir: Path, embed_cache_dir: Path, seed: int, dr_method: str, cluster_algo: str) -> Dict:
    data = load_jsonl(path)
    qa   = UnifiedQAModel(UQA_MODEL_NAME)
    sbert = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")
    cfg = build_ra_with_sbert_config(dr_method=dr_method, cluster_algo=cluster_algo)
    cache = RAPTORCache(cfg, tree_dir, seed)

    correct, n = 0, 0

    for i, ex in enumerate(tqdm(data, desc="QuALITY")):
        doc = _norm_text(ex["doc_text"])
        q   = _norm_text(ex["question"])
        choices = [_norm_text(c) for c in (ex.get("choices", []) or [])]
        gold_idx = ex.get("gold_idx", ex.get("correct_idx"))
        if gold_idx is None or not choices:
            continue
        n += 1

        doc_key = ex.get("article_id") or _sha1(doc)
        if with_raptor:
            ra = cache.get_or_build(doc, doc_key)
            ctx = ra.retrieve(q, top_k=RAPTOR_TOP_K, max_tokens=RETRIEVAL_BUDGET,
                              collapse_tree=True, return_layer_information=False)
        else:
            chunks = split_text(doc, TOK, LEAF_CHUNK_TOKENS)
            ctx = baseline_sbert_context_cached(
                chunks, q, sbert, TOK, embed_cache_dir, doc_key, RETRIEVAL_BUDGET
            )

        q_with_opts = _quality_prompt(q, choices)
        q_trim, c_trim = clip_for_unifiedqa(q_with_opts, _norm_text(ctx), budget=UQA_MAX_LEN)
        pred = qa.answer_question(c_trim, q_trim)
        guess = _parse_quality_pred(pred, choices)
        _maybe_debug_dump(i, q_trim, c_trim, pred, choices)
        correct += int(int(guess) == int(gold_idx))

    acc = 100.0 * correct / max(1, n)
    return {"accuracy": float(acc), "n": int(n)}

def run_qasper(path: Path, with_raptor: bool, tree_dir: Path, embed_cache_dir: Path, seed: int, dr_method: str, cluster_algo: str) -> Dict:
    data = load_jsonl(path)
    qa   = UnifiedQAModel(UQA_MODEL_NAME)
    sbert = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")
    cfg = build_ra_with_sbert_config(dr_method=dr_method, cluster_algo=cluster_algo)
    cache = RAPTORCache(cfg, tree_dir, seed)

    f1s, empties = [], 0

    for i, ex in enumerate(tqdm(data, desc="QASPER")):
        doc  = _norm_text(ex["doc_text"])
        q    = _norm_text(ex["question"])
        refs = [_norm_text(r) for r in (ex.get("gold_answers", []) or [])]
        doc_key = ex.get("paper_id") or _sha1(doc)

        if with_raptor:
            ra = cache.get_or_build(doc, doc_key)
            ctx = ra.retrieve(q, top_k=RAPTOR_TOP_K, max_tokens=RETRIEVAL_BUDGET,
                              collapse_tree=True, return_layer_information=False)
            if os.environ.get("DEBUG_EVAL") == "1":
                logging.info(f"[dbg] raw ctx tokens: {_tok_len_cl100k(_norm_text(ctx))}")
        else:
            chunks = split_text(doc, TOK, LEAF_CHUNK_TOKENS)
            ctx = baseline_sbert_context_cached(
                chunks, q, sbert, TOK, embed_cache_dir, doc_key, RETRIEVAL_BUDGET
            )

        q_trim, c_trim = clip_for_unifiedqa(q, _norm_text(ctx), budget=UQA_MAX_LEN)
        if os.environ.get("DEBUG_EVAL") == "1":
            logging.info(f"[dbg] clipped ctx tokens: {_tok_len_cl100k(c_trim)}")
            logging.info(f"[dbg] ctx sha1: {_sha1(c_trim)}")
        pred = qa.answer_question(c_trim, q_trim)

        if not pred.strip(): empties += 1
        _maybe_debug_dump(i, q_trim, c_trim, pred, refs)
        if refs:
            f1s.append(f1_answer(pred, refs))

    return {
        "f1": float(np.mean(f1s) * 100 if f1s else 0.0),
        "empty_preds": int(empties),
        "n": int(len(data)),
    }

# CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["narrativeqa", "quality", "qasper"], required=True)
    ap.add_argument("--split", required=True, help="Path to eval JSONL (e.g., .../eval_val.jsonl).")
    ap.add_argument("--with-raptor", action=argparse.BooleanOptionalAction, default=False,
                    help="Use RAPTOR (otherwise SBERT leaf-only baseline).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED],
                    help="One or more seeds to run. Example: --seeds 224 225 226")
    ap.add_argument("--out", default="results/table_runs.jsonl", help="Append JSONL results here.")
    ap.add_argument("--tree-dir", default="data/raptor_trees", help="Base dir to cache/load RAPTOR trees")
    ap.add_argument("--baseline-embeds", default="data/leaf_embeds", help="Base dir to cache leaf embeddings")
    ap.add_argument("--run-name", default="", help="Optional tag stored with results.")
    ap.add_argument("--dr-method", choices=["umap", "pacmap", "trimap", "none"], default="umap", help="Dimensionality reduction method used inside RAPTOR clustering.")
    ap.add_argument("--clusterer", choices=["raptor", "hdbscan"], default="raptor", help="Clustering algorithm for building trees: 'raptor' (GMM) or 'hdbscan'.")
    
    args = ap.parse_args()

    path = Path(args.split)
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path.resolve()}")

    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    for seed in args.seeds:
        set_global_seed(seed)
        seed_tag = f"seed{seed}"
        tree_dir = Path(args.tree_dir) / args.dataset / f"{seed_tag}_{args.dr_method}_{args.clusterer}"
        embed_cache_dir = Path(args.baseline_embeds) / seed_tag

        t0 = time.time()
        if args.dataset == "narrativeqa":
            metrics = run_narrativeqa(
                path, args.with_raptor, tree_dir, embed_cache_dir, seed, args.dr_method, args.clusterer
            )
        elif args.dataset == "quality":
            metrics = run_quality(
                path, args.with_raptor, tree_dir, embed_cache_dir, seed, args.dr_method, args.clusterer
            )
        else:
            metrics = run_qasper(
                path, args.with_raptor, tree_dir, embed_cache_dir, seed, args.dr_method, args.clusterer
            )

        elapsed = time.time() - t0

        record = {
            "timestamp": int(time.time()),
            "run_name": args.run_name or None,
            "dataset": args.dataset,
            "split": str(path),
            "with_raptor": bool(args.with_raptor),
            "seed": int(seed),
            "retrieval_budget": RETRIEVAL_BUDGET,
            "uqa_context_budget": UQA_CONTEXT_BUDGET,
            "tb_max_tokens": LEAF_CHUNK_TOKENS,
            "metrics": metrics,
            "elapsed_sec": round(elapsed, 2),
        }
        print(json.dumps(record, indent=2))
        append_jsonl(Path(args.out), record)

if __name__ == "__main__":
    main()
