#!/usr/bin/env python3
"""
Summary-quality vs QA correlation for RAPTOR trees.

For a given dataset / seed / DR method / clusterer, this script:

  1. Loads all RAPTOR trees from:
       <tree_root>/<dataset>/seed<seed>_<dr_method>_<clusterer>/<doc_key>.pkl

  2. For each document:
       - Computes summary quality stats from the tree:
         * parent–child cosine similarity (SBERT embeddings)
         * compression ratio = summary_tokens / sum(child_tokens)
         * layer-wise parent–child similarity

  3. Re-runs RAPTOR+UnifiedQA on that document's questions ONLY, and computes
     a single QA score per document:
       - NarrativeQA: mean ROUGE-L over its questions
       - QuALITY: mean accuracy over its questions (0/1)
       - QASPER: mean F1 over its questions

  4. Computes correlations across documents between:
       - doc-level mean parent–child similarity and QA score
       - doc-level mean compression ratio and QA score

  5. Saves all per-doc stats + correlations in a JSON file.

Usage (example):

  python analysis/summary_quality_vs_qa.py \
      --dataset quality \
      --split data/processed/quality/eval_val.jsonl \
      --tree-root data/raptor_trees \
      --seed 224 \
      --dr-method umap \
      --clusterer raptor \
      --out results/quality_summary_vs_qa.json
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import tiktoken
TOK = tiktoken.get_encoding("cl100k_base")

# --- Import shared constants & helpers from your eval script -----------------
# Adjust this import if your file is named differently (e.g. evaluation.eval)
from sbert_raptor_eval import (
    RETRIEVAL_BUDGET,
    UQA_MODEL_NAME,
    UQA_MAX_LEN,
    RAPTOR_TOP_K,
    _norm_text,
    _sha1,
    clip_for_unifiedqa,
    build_ra_with_sbert_config,
)

from raptor.RetrievalAugmentation import RetrievalAugmentation
from raptor.QAModels import UnifiedQAModel
from raptor.EmbeddingModels import SBertEmbeddingModel  # only used indirectly

# -----------------------------
# Basic IO helpers
# -----------------------------

def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_doc_key(dataset: str, ex: Dict, doc_text: str) -> str:
    """
    Mirror your eval logic for deriving doc_key.
    """
    if dataset == "narrativeqa":
        return ex.get("document_id") or _sha1(doc_text)
    elif dataset == "quality":
        return ex.get("article_id") or _sha1(doc_text)
    elif dataset == "qasper":
        return str(ex.get("paper_id") or ex.get("document_id") or _sha1(doc_text))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# -----------------------------
# Metric utilities
# -----------------------------

import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Make sure NLTK resources are available
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# BLEU / ROUGE / METEOR for NarrativeQA

def bleu1(refs: List[str], hyp: str) -> float:
    ref_tok = [word_tokenize(r) for r in refs]
    hyp_tok = word_tokenize(hyp)
    ch = SmoothingFunction().method1
    return sentence_bleu(ref_tok, hyp_tok, weights=(1.0, 0, 0, 0),
                         smoothing_function=ch) * 100


def bleu4_equal(refs: List[str], hyp: str) -> float:
    ref_tok = [word_tokenize(r) for r in refs]
    hyp_tok = word_tokenize(hyp)
    ch = SmoothingFunction().method1
    return sentence_bleu(ref_tok, hyp_tok, weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=ch) * 100


def rougeL(refs: List[str], hyp: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, hyp)["rougeL"].fmeasure for r in refs]
    return (max(scores) * 100) if scores else 0.0


def meteor_tokenized(refs: List[str], hyp: str) -> float:
    ref_tokens = [word_tokenize(r) for r in refs]
    hyp_tokens = word_tokenize(hyp)
    return meteor_score(ref_tokens, hyp_tokens) * 100


# SQuAD-style F1 for QASPER

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
        rec = common / len(B)
        return 2 * prec * rec / (prec + rec)

    return max((f1(pred, g) for g in golds), default=0.0)


# QuALITY multiple-choice helpers

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def quality_prompt(question: str, choices: List[str]) -> str:
    lettered = "\n".join([f"({chr(ord('A')+i)}) {c}" for i, c in enumerate(choices)])
    return f"{question}\n\nOptions:\n{lettered}\n\nAnswer with the option letter."


def parse_quality_pred(pred: str, choices: List[str]) -> int:
    p = pred.strip().upper()
    # Look for letter A/B/C/...
    for i, L in enumerate(_LETTERS[: len(choices)]):
        if re.search(rf"\b{L}\b", p):
            return i
    # OPTION X pattern
    m = re.search(r"\bOPTION\s+([A-Z])\b", p)
    if m:
        idx = ord(m.group(1)) - ord("A")
        if 0 <= idx < len(choices):
            return idx
    # Numeric index
    m = re.search(r"\b([0-9])\b", p)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(choices):
            return idx
    # Fallback: overlap with choice text
    p_low = pred.lower()
    sims = []
    for i, c in enumerate(choices):
        ctoks = set(word_tokenize(c.lower()))
        ptoks = set(word_tokenize(p_low))
        sims.append(len(ctoks & ptoks))
    return int(np.argmax(sims)) if sims else 0


# -----------------------------
# Cosine + aggregation helpers
# -----------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    num = float((a * b).sum())
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


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


# -----------------------------
# Tree loading and summary quality
# -----------------------------

def load_tree_ra(
    tree_root: Path,
    dataset: str,
    seed: int,
    dr_method: str,
    clusterer: str,
    doc_key: str,
    cfg_cache: Dict[str, Any],
) -> Tuple[RetrievalAugmentation, Any]:
    """
    Load a RAPTOR tree for a given doc_key.

    Assumes path layout:
      tree_root / dataset / seed<seed>_<dr_method>_<clusterer> / <doc_key>.pkl
    """
    seed_tag = f"seed{seed}"
    subdir = f"{seed_tag}_{dr_method}_{clusterer}"
    doc_base = tree_root / dataset / subdir / doc_key
    pkl_path = doc_base.with_suffix(".pkl")

    if not pkl_path.exists():
        raise FileNotFoundError(str(pkl_path))

    if "cfg" not in cfg_cache:
        # Use your existing helper to build a matching RetrievalAugmentationConfig.
        # We don't rebuild trees; we just need a retriever compatible with SBERT.
        cfg_cache["cfg"] = build_ra_with_sbert_config(
            dr_method=dr_method,
            cluster_algo=clusterer,
        )
    cfg = cfg_cache["cfg"]

    ra = RetrievalAugmentation(cfg, tree=str(pkl_path))
    return ra, ra.tree


def compute_summary_quality(tree: Any) -> Dict[str, Any]:
    """
    Compute summary quality statistics for a single tree:

      - parent–child cosine similarity (SBERT embeddings)
      - compression ratio = summary_tokens / child_tokens
      - layer-wise parent–child similarity

    Returns a dict with aggregated stats and counts.
    """
    parent_child_sims: List[float] = []
    compression_ratios: List[float] = []
    layer_sims: Dict[int, List[float]] = defaultdict(list)

    all_nodes = getattr(tree, "all_nodes", {})
    if not all_nodes:
        return {
            "parent_child_similarity": aggregate_stats([]),
            "compression_ratio": aggregate_stats([]),
            "layer_parent_child_similarity": {},
            "n_parents_used": 0,
        }

    for node_id, node in all_nodes.items():
        children = getattr(node, "children", None)
        if not children:
            continue  # skip leaves

        # Need SBERT embeddings for parent + children
        if not hasattr(node, "embeddings") or "SBERT" not in node.embeddings:
            continue

        parent_emb = np.asarray(node.embeddings["SBERT"], dtype=np.float32)
        if parent_emb.size == 0:
            continue

        parent_tokens = len(TOK.encode(node.text or ""))

        child_embs: List[np.ndarray] = []
        child_token_sum = 0
        for cid in children:
            child = all_nodes[cid]
            if not hasattr(child, "embeddings") or "SBERT" not in child.embeddings:
                continue
            ce = np.asarray(child.embeddings["SBERT"], dtype=np.float32)
            if ce.size == 0:
                continue
            child_embs.append(ce)
            child_token_sum += len(TOK.encode(child.text or ""))

        if not child_embs or child_token_sum == 0:
            continue

        # Compression ratio: parent summary length vs sum of children
        compression = parent_tokens / float(child_token_sum)
        compression_ratios.append(compression)

        # Parent–child cosine similarities
        layer = getattr(node, "layer", 0)
        for ce in child_embs:
            sim = cosine_sim(parent_emb, ce)
            parent_child_sims.append(sim)
            layer_sims[layer].append(sim)

    return {
        "parent_child_similarity": aggregate_stats(parent_child_sims),
        "compression_ratio": aggregate_stats(compression_ratios),
        "layer_parent_child_similarity": {
            int(layer): aggregate_stats(vals) for layer, vals in layer_sims.items()
        },
        "n_parents_used": len(compression_ratios),
    }


# -----------------------------
# QA evaluation per document
# -----------------------------

def compute_doc_level_scores(
    dataset: str,
    data: List[Dict],
    tree_root: Path,
    seed: int,
    dr_method: str,
    clusterer: str,
    max_docs: int | None = None,
    max_q_per_doc: int | None = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    For each document (doc_key), re-run RAPTOR+UnifiedQA and compute:

      doc_qa_scores[doc_key] = mean QA score over its questions
      doc_summary_stats[doc_key] = summary-quality stats for its tree

    Returns:
      (doc_qa_scores, doc_summary_stats)
    """

    # Group example indices by doc_key
    docs_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, ex in enumerate(data):
        doc_text = _norm_text(ex["doc_text"])
        doc_key = get_doc_key(dataset, ex, doc_text)
        docs_to_indices[doc_key].append(i)

    # UnifiedQA model (reader)
    qa = UnifiedQAModel(UQA_MODEL_NAME)

    cfg_cache: Dict[str, Any] = {}
    doc_qa_scores: Dict[str, float] = {}
    doc_summary_stats: Dict[str, Dict[str, Any]] = {}

    # Iterate over docs (optionally limiting count)
    all_doc_keys = list(docs_to_indices.keys())
    if max_docs is not None:
        all_doc_keys = all_doc_keys[: max_docs]

    for doc_idx, doc_key in enumerate(all_doc_keys):
        ex_indices = docs_to_indices[doc_key]
        if max_q_per_doc is not None:
            ex_indices = ex_indices[: max_q_per_doc]

        # Load tree (and RA for retrieval)
        try:
            ra, tree = load_tree_ra(
                tree_root=tree_root,
                dataset=dataset,
                seed=seed,
                dr_method=dr_method,
                clusterer=clusterer,
                doc_key=doc_key,
                cfg_cache=cfg_cache,
            )
        except FileNotFoundError:
            logging.warning(f"[skip] No tree for doc_key='{doc_key}'")
            continue

        # Compute summary quality once per doc
        summary_stats = compute_summary_quality(tree)
        doc_summary_stats[doc_key] = summary_stats

        per_q_scores: List[float] = []

        for ex_idx in ex_indices:
            ex = data[ex_idx]
            doc_text = _norm_text(ex["doc_text"])
            question = _norm_text(ex["question"])

            if dataset == "narrativeqa":
                refs = [_norm_text(r) for r in (ex.get("gold_answers", []) or [])]
                if not refs:
                    continue

                ctx = ra.retrieve(
                    question,
                    top_k=RAPTOR_TOP_K,
                    max_tokens=RETRIEVAL_BUDGET,
                    collapse_tree=True,
                    return_layer_information=False,
                )
                q_trim, c_trim = clip_for_unifiedqa(question, _norm_text(ctx), budget=UQA_MAX_LEN)
                pred = qa.answer_question(c_trim, q_trim)
                score = rougeL(refs, pred)

            elif dataset == "quality":
                choices = [_norm_text(c) for c in (ex.get("choices", []) or [])]
                gold_idx = ex.get("gold_idx", ex.get("correct_idx"))
                if gold_idx is None or not choices:
                    continue

                q_with_opts = quality_prompt(question, choices)
                ctx = ra.retrieve(
                    q_with_opts,
                    top_k=RAPTOR_TOP_K,
                    max_tokens=RETRIEVAL_BUDGET,
                    collapse_tree=True,
                    return_layer_information=False,
                )
                q_trim, c_trim = clip_for_unifiedqa(q_with_opts, _norm_text(ctx), budget=UQA_MAX_LEN)
                pred = qa.answer_question(c_trim, q_trim)
                guess = parse_quality_pred(pred, choices)
                score = 100.0 * float(int(guess) == int(gold_idx))

            elif dataset == "qasper":
                refs = [_norm_text(r) for r in (ex.get("gold_answers", []) or [])]
                if not refs:
                    continue

                ctx = ra.retrieve(
                    question,
                    top_k=RAPTOR_TOP_K,
                    max_tokens=RETRIEVAL_BUDGET,
                    collapse_tree=True,
                    return_layer_information=False,
                )
                q_trim, c_trim = clip_for_unifiedqa(question, _norm_text(ctx), budget=UQA_MAX_LEN)
                pred = qa.answer_question(c_trim, q_trim)
                score = f1_answer(pred, refs) * 100.0

            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

            per_q_scores.append(score)

        if not per_q_scores:
            logging.warning(f"[doc-level] No questions scored for doc_key='{doc_key}'")
            continue

        doc_score = float(np.mean(per_q_scores))
        doc_qa_scores[doc_key] = doc_score

        logging.info(
            f"[{doc_idx+1}/{len(all_doc_keys)}] doc_key={doc_key} "
            f"QA={doc_score:.2f}, parents_used={summary_stats['n_parents_used']}"
        )

    return doc_qa_scores, doc_summary_stats


# -----------------------------
# Correlation computation
# -----------------------------

def compute_correlations(
    doc_qa_scores: Dict[str, float],
    doc_summary_stats: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute Pearson correlations across docs between:

      - QA score and mean parent–child similarity
      - QA score and mean compression ratio
    """
    common_keys = sorted(set(doc_qa_scores.keys()) & set(doc_summary_stats.keys()))
    if len(common_keys) < 2:
        logging.warning("Not enough documents to compute correlations.")
        return {
            "n_docs_overlap": len(common_keys),
            "qa_vs_parent_child_sim_mean": 0.0,
            "qa_vs_compression_ratio_mean": 0.0,
        }

    parent_child_means: List[float] = []
    compression_means: List[float] = []
    qa_vals: List[float] = []

    for k in common_keys:
        pc = doc_summary_stats[k]["parent_child_similarity"]["mean"]
        cr = doc_summary_stats[k]["compression_ratio"]["mean"]
        qv = doc_qa_scores[k]
        parent_child_means.append(pc)
        compression_means.append(cr)
        qa_vals.append(qv)

    pc_arr = np.array(parent_child_means, dtype=np.float32)
    cr_arr = np.array(compression_means, dtype=np.float32)
    qa_arr = np.array(qa_vals, dtype=np.float32)

    def pearson(a, b):
        if a.size < 2 or b.size < 2:
            return 0.0
        c = np.corrcoef(a, b)
        return float(c[0, 1])

    return {
        "n_docs_overlap": len(common_keys),
        "qa_vs_parent_child_sim_mean": pearson(qa_arr, pc_arr),
        "qa_vs_compression_ratio_mean": pearson(qa_arr, cr_arr),
    }


# -----------------------------
# Main CLI
# -----------------------------

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
        "--tree-root",
        required=True,
        help="Base dir where RAPTOR trees live "
             "(will look under <tree-root>/<dataset>/seed<seed>_<dr>_<clusterer>/).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=224,
        help="Seed used for tree building (e.g., 224).",
    )
    ap.add_argument(
        "--dr-method",
        choices=["umap", "pacmap", "trimap", "none"],
        default="umap",
        help="DR method used when building the trees (for locating the right dir).",
    )
    ap.add_argument(
        "--clusterer",
        choices=["raptor", "hdbscan"],
        default="raptor",
        help="Clustering algorithm used when building the trees (for locating the right dir).",
    )
    ap.add_argument(
        "--out",
        default="results/summary_quality_vs_qa.json",
        help="Output JSON file.",
    )
    ap.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional: limit number of documents (for debugging).",
    )
    ap.add_argument(
        "--max-q-per-doc",
        type=int,
        default=None,
        help="Optional: limit number of questions per doc (for debugging).",
    )

    args = ap.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    dataset = args.dataset
    split_path = Path(args.split)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    tree_root = Path(args.tree_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading data from {split_path} for dataset={dataset}")
    data = load_jsonl(split_path)

    logging.info("Computing doc-level QA scores and summary quality stats ...")
    doc_qa_scores, doc_summary_stats = compute_doc_level_scores(
        dataset=dataset,
        data=data,
        tree_root=tree_root,
        seed=args.seed,
        dr_method=args.dr_method,
        clusterer=args.clusterer,
        max_docs=args.max_docs,
        max_q_per_doc=args.max_q_per_doc,
    )

    logging.info(
        f"Got QA scores for {len(doc_qa_scores)} docs, "
        f"summary stats for {len(doc_summary_stats)} docs."
    )

    logging.info("Computing correlations ...")
    corr_stats = compute_correlations(doc_qa_scores, doc_summary_stats)

    # Aggregate some global summary-quality stats over docs (means of means)
    pc_means = [v["parent_child_similarity"]["mean"] for v in doc_summary_stats.values()]
    cr_means = [v["compression_ratio"]["mean"] for v in doc_summary_stats.values()]

    global_summary_stats = {
        "parent_child_similarity_mean_over_docs": aggregate_stats(pc_means),
        "compression_ratio_mean_over_docs": aggregate_stats(cr_means),
    }

    result = {
        "dataset": dataset,
        "seed": int(args.seed),
        "dr_method": args.dr_method,
        "clusterer": args.clusterer,
        "n_docs_with_qa": len(doc_qa_scores),
        "n_docs_with_summary_stats": len(doc_summary_stats),
        "doc_scores": doc_qa_scores,          # doc_key -> QA score
        "doc_summary_stats": doc_summary_stats,  # doc_key -> summary quality stats
        "global_summary_stats": global_summary_stats,
        "correlations": corr_stats,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logging.info(f"Wrote summary-quality vs QA analysis to {out_path}")


if __name__ == "__main__":
    main()


#### HAVEN'T RUN THIS ONE YET!