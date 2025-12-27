#!/usr/bin/env python3
"""
Post-hoc clustering + summarization baseline (NO TREE).

This file is the same post-hoc baseline you had (HDBSCAN only), but extended to:
  - support NarrativeQA + QuALITY + QASPER
  - compute metrics IDENTICALLY to your RAPTOR eval:
      NarrativeQA: BLEU-1, BLEU-4(equal), ROUGE-L(F1+stemmer), METEOR(tokenized)
      QuALITY: accuracy with the same prompt + parsing
      QASPER: SQuAD-style token-F1
  - use the same UnifiedQA clipping logic as RAPTOR eval (budget + safety + 400 ctx tokens)

It still:
  - builds leaf chunks with split_text(100 tokens, cl100k_base)
  - baseline retrieves with FAISS IndexFlatIP + greedy token budget (2000)
  - posthoc summarizes the HDBSCAN clusters touched by baseline retrieval
  - prepends top-K touched cluster summaries to baseline context

Fixes in this version:
  - Clip summarization input to avoid OpenAI context length errors (NarrativeQA has huge docs).
  - Never store exception objects as summaries; skip failed summaries safely.
"""

import argparse
import json
import logging
import os
import re
import string
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tiktoken
import hdbscan
import faiss
from tqdm import tqdm

# -------------------------
# Determinism (same style)
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

DEFAULT_SEED = 224

# -------------------------
# Shared constants (paper-aligned)
# -------------------------
LEAF_CHUNK_TOKENS = 100
RETRIEVAL_BUDGET = 2000
UQA_MODEL_NAME = "allenai/unifiedqa-v2-t5-3b-1363200"
UQA_MAX_LEN = 512
UQA_CONTEXT_BUDGET = 400
UQA_SAFETY = 12

SUMMARY_MAX_TOKENS = 100
CLUSTER_TOP_K = 5          # mirror RAPTOR TreeBuilder top_k=5
BASELINE_TOP_K = 50        # mirror RAPTOR retriever top_k usage

# NEW: summarizer input safety cap (GPT-3.5-turbo often 16k context; keep margin)
SUMMARIZER_MAX_INPUT_TOKENS = 12000

TOK = tiktoken.get_encoding("cl100k_base")

# -------------------------
# Models
# -------------------------
from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor.QAModels import UnifiedQAModel
from raptor.SummarizationModels import GPT3TurboSummarizationModel
from raptor.utils import split_text

_SBERT = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# -------------------------
# IO helpers
# -------------------------
def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

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

# -------------------------
# UnifiedQA clipping (IDENTICAL to your RAPTOR eval)
# -------------------------
from transformers import AutoTokenizer
_UQA_TOK = AutoTokenizer.from_pretrained(UQA_MODEL_NAME, use_fast=False)
if not getattr(_UQA_TOK, "model_max_length", None):
    _UQA_TOK.model_max_length = UQA_MAX_LEN

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

# -------------------------
# Metrics (IDENTICAL to your RAPTOR eval)
# -------------------------
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

# QASPER token-F1 (SQuAD-style)
_ARTS = {"a", "an", "the"}
_PUNC_TABLE = str.maketrans("", "", string.punctuation)

def _normalize_f1(s: str) -> List[str]:
    s = s.lower()
    s = s.translate(_PUNC_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split()
    return [t for t in toks if t not in _ARTS]

def f1_answer(pred: str, golds: List[str]) -> float:
    def f1(a, b):
        A, B = _normalize_f1(a), _normalize_f1(b)
        if not A or not B:
            return 0.0
        common = 0
        for w in set(A + B):
            common += min(A.count(w), B.count(w))
        if common == 0:
            return 0.0
        prec = common / len(A)
        rec  = common / len(B)
        return 2 * prec * rec / (prec + rec)
    return max((f1(pred, g) for g in golds), default=0.0)

# QuALITY prompting/parsing (IDENTICAL)
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
        if 0 <= i < len(choices):
            return i
    m = re.search(r"\b([0-9])\b", p)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(choices):
            return idx
    # fallback: overlap
    p_low = pred.lower()
    sims = []
    for i, c in enumerate(choices):
        ctoks = set(word_tokenize(c.lower()))
        ptoks = set(word_tokenize(p_low))
        sims.append(len(ctoks & ptoks))
    return int(np.argmax(sims)) if sims else 0

# -------------------------
# Baseline retrieval (FAISS IP + greedy budget)
# -------------------------
def retrieve_baseline(embs: np.ndarray, texts: List[str], q: str):
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    qv = np.array([_SBERT.create_embedding(_norm_text(q))], dtype=np.float32)
    _, I = index.search(qv, min(BASELINE_TOP_K, len(texts)))

    toks, parts, idxs = 0, [], []
    for i in I[0]:
        t = _norm_text(texts[int(i)])
        nt = len(TOK.encode(t))
        if toks + nt > RETRIEVAL_BUDGET:
            break
        toks += nt
        parts.append(t)
        idxs.append(int(i))
    return "\n\n".join(parts), idxs

# -------------------------
# Post-hoc clustering (HDBSCAN only)
# -------------------------
def cluster_hdbscan(embs: np.ndarray) -> np.ndarray:
    if embs.shape[0] == 0:
        return np.array([], dtype=int)
    if embs.shape[0] == 1:
        return np.array([0], dtype=int)

    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean"
    ).fit_predict(X)

    # Map noise to singleton clusters
    labels = labels.astype(int)
    next_id = labels.max() + 1 if labels.size else 0
    for i, l in enumerate(labels):
        if l == -1:
            labels[i] = next_id
            next_id += 1
    return labels.astype(int)

# -------------------------
# NEW: summarizer safety helpers
# -------------------------
def clip_for_summarizer(text: str, max_tokens: int = SUMMARIZER_MAX_INPUT_TOKENS) -> str:
    ids = TOK.encode(text)
    if len(ids) <= max_tokens:
        return text
    return TOK.decode(ids[:max_tokens])

def safe_summarize(summarizer, text: str, max_tokens: int) -> str | None:
    try:
        out = summarizer.summarize(text, max_tokens=max_tokens)
        if isinstance(out, str) and out.strip():
            return out
        return None
    except Exception as e:
        logging.warning(f"Summarization failed (skipping): {type(e).__name__}: {e}")
        return None

# -------------------------
# doc key helper (stable across datasets)
# -------------------------
def get_doc_key(dataset: str, ex: Dict, doc_text: str) -> str:
    if dataset == "narrativeqa":
        return ex.get("document_id") or _sha1(doc_text)
    if dataset == "quality":
        return ex.get("article_id") or _sha1(doc_text)
    if dataset == "qasper":
        return str(ex.get("paper_id") or ex.get("document_id") or _sha1(doc_text))
    raise ValueError(dataset)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["narrativeqa", "quality", "qasper"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Random seed (reproducibility for HDBSCAN + summarization)")
    ap.add_argument("--cluster-top-k", type=int, default=CLUSTER_TOP_K)
    ap.add_argument("--summary-max-tokens", type=int, default=SUMMARY_MAX_TOKENS)
    ap.add_argument("--summarizer-max-input-tokens", type=int, default=SUMMARIZER_MAX_INPUT_TOKENS)
    args = ap.parse_args()

    summarizer_max_input_tokens = int(args.summarizer_max_input_tokens)


    set_global_seed(args.seed)

    data = load_jsonl(Path(args.split))
    qa = UnifiedQAModel(UQA_MODEL_NAME)
    summarizer = GPT3TurboSummarizationModel("gpt-3.5-turbo")

    # Group by document (avoid recomputing clusters/summaries)
    doc_texts: Dict[str, str] = {}
    for ex in data:
        doc = _norm_text(ex["doc_text"])
        dk = get_doc_key(args.dataset, ex, doc)
        doc_texts.setdefault(dk, doc)

    # Precompute chunks + embeddings + HDBSCAN + summaries per doc
    doc_chunks: Dict[str, List[str]] = {}
    doc_embs: Dict[str, np.ndarray] = {}
    doc_labels: Dict[str, np.ndarray] = {}
    doc_summaries: Dict[str, Dict[int, str]] = {}

    for dk, doc in tqdm(doc_texts.items(), desc="Precompute docs"):
        chunks = split_text(doc, TOK, LEAF_CHUNK_TOKENS)
        if not chunks:
            continue

        embs = np.array([_SBERT.create_embedding(_norm_text(c)) for c in chunks], dtype=np.float32)
        labels = cluster_hdbscan(embs)

        summaries: Dict[int, str] = {}
        for cid in sorted(set(labels.tolist())):
            idxs = np.where(labels == cid)[0]
            if len(idxs) <= 1:
                continue  # NO singleton summaries

            text = "\n\n".join(chunks[i] for i in idxs)
            text = clip_for_summarizer(text, max_tokens=summarizer_max_input_tokens)

            summ = safe_summarize(summarizer, text, max_tokens=int(args.summary_max_tokens))
            if summ:
                summaries[int(cid)] = summ

        doc_chunks[dk] = chunks
        doc_embs[dk] = embs
        doc_labels[dk] = labels
        doc_summaries[dk] = summaries

    # Accumulators (match RAPTOR metrics)
    if args.dataset == "narrativeqa":
        base = {"bleu1": [], "bleu4": [], "rougeL": [], "meteor": []}
        post = {"bleu1": [], "bleu4": [], "rougeL": [], "meteor": []}
        empties_base = 0
        empties_post = 0

    elif args.dataset == "quality":
        base_correct = 0
        post_correct = 0
        n_used = 0

    else:  # qasper
        base_f1s: List[float] = []
        post_f1s: List[float] = []
        empties_base = 0
        empties_post = 0

    # Per-example loop
    for ex in tqdm(data, desc=f"Eval {args.dataset}"):
        doc = _norm_text(ex["doc_text"])
        q = _norm_text(ex["question"])
        dk = get_doc_key(args.dataset, ex, doc)

        if dk not in doc_chunks:
            continue

        # Dataset-specific retrieval question
        if args.dataset == "quality":
            choices = [_norm_text(c) for c in (ex.get("choices", []) or [])]
            gold_idx = ex.get("gold_idx", ex.get("correct_idx"))
            if gold_idx is None or not choices:
                continue
            q_for_retrieval = _quality_prompt(q, choices)
        else:
            q_for_retrieval = q

        # 1) baseline retrieval
        base_ctx, idxs = retrieve_baseline(doc_embs[dk], doc_chunks[dk], q_for_retrieval)
        if not idxs:
            continue

        # 2) clusters touched by baseline -> rank by query similarity -> take summaries
        used_clusters = sorted(set(int(doc_labels[dk][i]) for i in idxs))

        qv = _SBERT.create_embedding(_norm_text(q_for_retrieval)).astype(np.float32)
        scored = []
        for cid in used_clusters:
            if cid not in doc_summaries[dk]:
                continue
            members = np.where(doc_labels[dk] == cid)[0]
            score = float(np.max(doc_embs[dk][members] @ qv))
            scored.append((score, cid))

        scored.sort(reverse=True)
        top_summaries_raw = [doc_summaries[dk][cid] for _, cid in scored[:int(args.cluster_top_k)]]
        top_summaries = [s for s in top_summaries_raw if isinstance(s, str) and s.strip()]

        post_ctx = "\n\n".join(top_summaries + [base_ctx]) if top_summaries else base_ctx

        # 3) Run UnifiedQA with the SAME clipping logic as RAPTOR eval
        if args.dataset == "narrativeqa":
            refs = [_norm_text(r) for r in (ex.get("gold_answers", []) or [])]
            if not refs:
                continue

            q_b, c_b = clip_for_unifiedqa(q, base_ctx, budget=UQA_MAX_LEN)
            pred_b = qa.answer_question(c_b, q_b)
            if not pred_b.strip():
                empties_base += 1

            q_p, c_p = clip_for_unifiedqa(q, post_ctx, budget=UQA_MAX_LEN)
            pred_p = qa.answer_question(c_p, q_p)
            if not pred_p.strip():
                empties_post += 1

            base["bleu1"].append(bleu1(refs, pred_b))
            base["bleu4"].append(bleu4_equal(refs, pred_b))
            base["rougeL"].append(rougeL(refs, pred_b))
            base["meteor"].append(meteor_tokenized(refs, pred_b))

            post["bleu1"].append(bleu1(refs, pred_p))
            post["bleu4"].append(bleu4_equal(refs, pred_p))
            post["rougeL"].append(rougeL(refs, pred_p))
            post["meteor"].append(meteor_tokenized(refs, pred_p))

        elif args.dataset == "quality":
            choices = [_norm_text(c) for c in (ex.get("choices", []) or [])]
            gold_idx = ex.get("gold_idx", ex.get("correct_idx"))
            if gold_idx is None or not choices:
                continue

            q_trim_b, c_trim_b = clip_for_unifiedqa(q_for_retrieval, base_ctx, budget=UQA_MAX_LEN)
            pred_b = qa.answer_question(c_trim_b, q_trim_b)
            guess_b = _parse_quality_pred(pred_b, choices)

            q_trim_p, c_trim_p = clip_for_unifiedqa(q_for_retrieval, post_ctx, budget=UQA_MAX_LEN)
            pred_p = qa.answer_question(c_trim_p, q_trim_p)
            guess_p = _parse_quality_pred(pred_p, choices)

            base_correct += int(int(guess_b) == int(gold_idx))
            post_correct += int(int(guess_p) == int(gold_idx))
            n_used += 1

        else:  # qasper
            refs = [_norm_text(r) for r in (ex.get("gold_answers", []) or [])]
            if not refs:
                continue

            q_b, c_b = clip_for_unifiedqa(q, base_ctx, budget=UQA_MAX_LEN)
            pred_b = qa.answer_question(c_b, q_b)
            if not pred_b.strip():
                empties_base += 1
            base_f1s.append(f1_answer(pred_b, refs))

            q_p, c_p = clip_for_unifiedqa(q, post_ctx, budget=UQA_MAX_LEN)
            pred_p = qa.answer_question(c_p, q_p)
            if not pred_p.strip():
                empties_post += 1
            post_f1s.append(f1_answer(pred_p, refs))

    # Aggregate + write output (table-friendly)
    if args.dataset == "narrativeqa":
        result = {
            "timestamp": int(time.time()),
            "dataset": args.dataset,
            "split": str(Path(args.split)),
            "seed": int(args.seed),
            "retrieval_budget": RETRIEVAL_BUDGET,
            "uqa_context_budget": UQA_CONTEXT_BUDGET,
            "tb_max_tokens": LEAF_CHUNK_TOKENS,
            "summary_max_tokens": int(args.summary_max_tokens),
            "cluster_top_k": int(args.cluster_top_k),
            "summarizer_max_input_tokens": int(summarizer_max_input_tokens),
            "metrics": {
                "baseline": {
                    "bleu1": float(np.mean(base["bleu1"]) if base["bleu1"] else 0.0),
                    "bleu4": float(np.mean(base["bleu4"]) if base["bleu4"] else 0.0),
                    "rougeL": float(np.mean(base["rougeL"]) if base["rougeL"] else 0.0),
                    "meteor": float(np.mean(base["meteor"]) if base["meteor"] else 0.0),
                    "empty_preds": int(empties_base),
                    "n": int(len(base["bleu1"])),
                },
                "posthoc_cluster": {
                    "bleu1": float(np.mean(post["bleu1"]) if post["bleu1"] else 0.0),
                    "bleu4": float(np.mean(post["bleu4"]) if post["bleu4"] else 0.0),
                    "rougeL": float(np.mean(post["rougeL"]) if post["rougeL"] else 0.0),
                    "meteor": float(np.mean(post["meteor"]) if post["meteor"] else 0.0),
                    "empty_preds": int(empties_post),
                    "n": int(len(post["bleu1"])),
                },
            },
        }

    elif args.dataset == "quality":
        acc_base = 100.0 * base_correct / max(1, n_used)
        acc_post = 100.0 * post_correct / max(1, n_used)
        result = {
            "timestamp": int(time.time()),
            "dataset": args.dataset,
            "split": str(Path(args.split)),
            "seed": int(args.seed),
            "retrieval_budget": RETRIEVAL_BUDGET,
            "uqa_context_budget": UQA_CONTEXT_BUDGET,
            "tb_max_tokens": LEAF_CHUNK_TOKENS,
            "summary_max_tokens": int(args.summary_max_tokens),
            "cluster_top_k": int(args.cluster_top_k),
            "summarizer_max_input_tokens": int(SUMMARIZER_MAX_INPUT_TOKENS),
            "metrics": {
                "baseline": {"accuracy": float(acc_base), "n": int(n_used)},
                "posthoc_cluster": {"accuracy": float(acc_post), "n": int(n_used)},
            },
        }

    else:  # qasper
        result = {
            "timestamp": int(time.time()),
            "dataset": args.dataset,
            "split": str(Path(args.split)),
            "seed": int(args.seed),
            "retrieval_budget": RETRIEVAL_BUDGET,
            "uqa_context_budget": UQA_CONTEXT_BUDGET,
            "tb_max_tokens": LEAF_CHUNK_TOKENS,
            "summary_max_tokens": int(args.summary_max_tokens),
            "cluster_top_k": int(args.cluster_top_k),
            "summarizer_max_input_tokens": int(SUMMARIZER_MAX_INPUT_TOKENS),
            "metrics": {
                "baseline": {
                    "f1": float(np.mean(base_f1s) * 100 if base_f1s else 0.0),
                    "empty_preds": int(empties_base),
                    "n": int(len(base_f1s)),
                },
                "posthoc_cluster": {
                    "f1": float(np.mean(post_f1s) * 100 if post_f1s else 0.0),
                    "empty_preds": int(empties_post),
                    "n": int(len(post_f1s)),
                },
            },
        }

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    logging.info(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
