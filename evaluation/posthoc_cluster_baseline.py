#!/usr/bin/env python3
"""
Outputs metrics comparable to your other runs:
  - NarrativeQA: BLEU-1, BLEU-4, ROUGE-L, METEOR
  - QuALITY: accuracy
  - QASPER: token-level F1

  python analysis/posthoc_cluster_baseline.py \
      --dataset quality \
      --split data/processed/quality/eval_val_sub50_q5.jsonl \
      --seed 224 \
      --out results/quality_posthoc_cluster.json
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tiktoken
import hdbscan

from eval import (  # adjust if your file is named differently
    LEAF_CHUNK_TOKENS,
    RETRIEVAL_BUDGET,
    UQA_MODEL_NAME,
    UQA_MAX_LEN,
    _norm_text,
    _sha1,
    clip_for_unifiedqa,
)

from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor.QAModels import UnifiedQAModel
from raptor.SummarizationModels import GPT3TurboSummarizationModel
from raptor.utils import split_text

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

TOK = tiktoken.get_encoding("cl100k_base")

# Single global SBERT model to avoid re-loading
_GLOBAL_SBERT = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")


# Basic IO helpers

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
    Mirror your other analysis scripts for doc_key logic.
    """
    if dataset == "narrativeqa":
        return ex.get("document_id") or _sha1(doc_text)
    elif dataset == "quality":
        return ex.get("article_id") or _sha1(doc_text)
    elif dataset == "qasper":
        return str(ex.get("paper_id") or ex.get("document_id") or _sha1(doc_text))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# Metrics (copied / aligned with your other analysis scripts)

import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Ensure NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def bleu1(refs: List[str], hyp: str) -> float:
    ref_tok = [word_tokenize(r) for r in refs]
    hyp_tok = word_tokenize(hyp)
    ch = SmoothingFunction().method1
    return sentence_bleu(
        ref_tok, hyp_tok, weights=(1.0, 0, 0, 0), smoothing_function=ch
    ) * 100


def bleu4_equal(refs: List[str], hyp: str) -> float:
    ref_tok = [word_tokenize(r) for r in refs]
    hyp_tok = word_tokenize(hyp)
    ch = SmoothingFunction().method1
    return sentence_bleu(
        ref_tok, hyp_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=ch
    ) * 100


def rougeL(refs: List[str], hyp: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, hyp)["rougeL"].fmeasure for r in refs]
    return (max(scores) * 100) if scores else 0.0


def meteor_tokenized(refs: List[str], hyp: str) -> float:
    ref_tokens = [word_tokenize(r) for r in refs]
    hyp_tokens = word_tokenize(hyp)
    return meteor_score(ref_tokens, hyp_tokens) * 100


# QASPER F1

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
    lettered = "\n".join(
        [f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)]
    )
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


# SBERT retrieval & clustering

def baseline_retrieve_with_indices(
    leaf_embs: np.ndarray,
    leaf_texts: List[str],
    question: str,
    max_tokens: int = RETRIEVAL_BUDGET,
    top_k: int = 50,
) -> Tuple[int, List[int], str]:
    """
    SBERT baseline retrieval:
      - FAISS inner-product index over leaf_embs
      - greedy add under max_tokens budget
    Returns:
      total_tokens, indices, context_string
    """
    import faiss  # local import to avoid global dependency if unused

    if leaf_embs.size == 0 or not leaf_texts:
        return 0, [], ""

    index = faiss.IndexFlatIP(leaf_embs.shape[1])
    index.add(leaf_embs.astype(np.float32))

    q_vec = np.array([_GLOBAL_SBERT.create_embedding(question)], dtype=np.float32)

    k = min(len(leaf_texts), top_k)
    _, I = index.search(q_vec, k)

    total_tokens = 0
    picked: List[int] = []
    pieces: List[str] = []

    for raw_idx in I[0]:
        idx = int(raw_idx)
        piece = leaf_texts[idx]
        piece_tokens = len(TOK.encode(piece))
        if total_tokens + piece_tokens > max_tokens:
            break
        total_tokens += piece_tokens
        picked.append(idx)
        pieces.append(piece)

    context = "\n\n".join(pieces)
    return total_tokens, picked, context


def cluster_leaf_embeddings_hdbscan(
    leaf_embs: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 1,
) -> np.ndarray:
    """
    Run HDBSCAN on SBERT embeddings.
    Returns: labels (shape: [n_leaves]).
      - noise points get label -1; we map each such point to its own singleton cluster.
    """
    if leaf_embs.shape[0] == 0:
        return np.array([], dtype=int)
    if leaf_embs.shape[0] == 1:
        return np.array([0], dtype=int)

    # Normalize for cosine-ish behavior
    norms = np.linalg.norm(leaf_embs, axis=1, keepdims=True) + 1e-8
    X = leaf_embs / norms

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)

    # Map noise (-1) to unique cluster ids
    next_cluster = labels.max() + 1 if labels.size > 0 else 0
    labels_mapped = labels.copy()
    for i, lab in enumerate(labels):
        if lab == -1:
            labels_mapped[i] = next_cluster
            next_cluster += 1

    return labels_mapped.astype(int)


def build_clusters_and_summaries_for_doc(
    leaf_texts: List[str],
    leaf_embs: np.ndarray,
    summarizer: GPT3TurboSummarizationModel,
    summary_max_tokens: int = 100,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    For a single document:
      - cluster leaf embeddings
      - summarize each cluster once
    Returns:
      cluster_labels: array shape (n_leaves,)
      cluster_summaries: dict cluster_id -> summary text
    """
    if not leaf_texts:
        return np.array([], dtype=int), {}

    labels = cluster_leaf_embeddings_hdbscan(leaf_embs)
    cluster_summaries: Dict[int, str] = {}

    # indices per cluster
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        clusters[int(lab)].append(i)

    logging.info(f"  Found {len(clusters)} clusters for doc (incl. singletons).")

    for cid, idxs in clusters.items():
        # Concatenate texts in this cluster
        cluster_text_parts = [leaf_texts[i] for i in idxs]
        cluster_text = "\n\n".join(cluster_text_parts)
        summary = summarizer.summarize(cluster_text, max_tokens=summary_max_tokens)
        cluster_summaries[cid] = summary

    return labels, cluster_summaries


# Main experiment logic

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
        help="Path to eval JSONL (e.g. .../eval_val.jsonl or sub50_q5).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=224,
        help="Random seed (used only for reproducibility in downstream tweaks).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSON file for metrics.",
    )
    ap.add_argument(
        "--summary-max-tokens",
        type=int,
        default=100,  # matches TreeBuilder summarization_length in your main RAPTOR runs
        help="Max tokens for each cluster summary (should match TreeBuilder summarization_length).",
    )
    args = ap.parse_args()

    dataset = args.dataset
    split_path = Path(args.split)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading split: {split_path} (dataset={dataset})")
    data = load_jsonl(split_path)

    # Group example indices by doc_key
    docs_to_indices: Dict[str, List[int]] = defaultdict(list)
    doc_texts: Dict[str, str] = {}
    for i, ex in enumerate(data):
        doc_text = _norm_text(ex["doc_text"])
        doc_key = get_doc_key(dataset, ex, doc_text)
        docs_to_indices[doc_key].append(i)
        if doc_key not in doc_texts:
            doc_texts[doc_key] = doc_text

    logging.info(f"Found {len(docs_to_indices)} unique documents in split.")

    # Prepare models
    qa = UnifiedQAModel(UQA_MODEL_NAME)
    summarizer = GPT3TurboSummarizationModel(model="gpt-3.5-turbo")

    # Precompute per-doc chunks, embeddings, clusters, summaries
    doc_chunks: Dict[str, List[str]] = {}
    doc_embs: Dict[str, np.ndarray] = {}
    doc_cluster_labels: Dict[str, np.ndarray] = {}
    doc_cluster_summaries: Dict[str, Dict[int, str]] = {}

    for di, (doc_key, idxs) in enumerate(docs_to_indices.items(), start=1):
        doc_text = doc_texts[doc_key]
        # Leaf chunks as in your tree-building: 100-token chunks with cl100k_base
        chunks = split_text(doc_text, TOK, LEAF_CHUNK_TOKENS)
        if not chunks:
            logging.warning(f"[doc {doc_key}] has no chunks, skipping.")
            continue

        # SBERT embeddings for leaves
        embs = np.array(
            [_GLOBAL_SBERT.create_embedding(ch) for ch in chunks],
            dtype=np.float32,
        )

        # Clusters + summaries
        logging.info(
            f"[{di}/{len(docs_to_indices)}] Building clusters & summaries for doc_key={doc_key} "
            f"(n_chunks={len(chunks)})"
        )
        labels, summaries = build_clusters_and_summaries_for_doc(
            chunks,
            embs,
            summarizer=summarizer,
            summary_max_tokens=args.summary_max_tokens,
        )

        doc_chunks[doc_key] = chunks
        doc_embs[doc_key] = embs
        doc_cluster_labels[doc_key] = labels
        doc_cluster_summaries[doc_key] = summaries

    # ------------------------------------------------------------------
    # Run QA for baseline vs post-hoc per example
    # ------------------------------------------------------------------

    # Metric accumulators
    # NarrativeQA
    base_bleu1, base_bleu4, base_rougeL, base_meteor = [], [], [], []
    ph_bleu1, ph_bleu4, ph_rougeL, ph_meteor = [], [], [], []

    # QuALITY
    base_acc, ph_acc = [], []

    # QASPER
    base_f1, ph_f1 = [], []

    n_examples_used = 0

    for ex_idx, ex in enumerate(data):
        doc_text = _norm_text(ex["doc_text"])
        question = _norm_text(ex["question"])
        doc_key = get_doc_key(dataset, ex, doc_text)

        if doc_key not in doc_chunks:
            logging.warning(f"[ex {ex_idx}] No chunks for doc_key={doc_key}, skipping.")
            continue

        leaf_texts = doc_chunks[doc_key]
        leaf_embs = doc_embs[doc_key]
        labels = doc_cluster_labels[doc_key]
        summaries = doc_cluster_summaries[doc_key]

        # Build the retrieval question (for QuALITY we use Q+options)
        if dataset == "quality":
            choices = [_norm_text(c) for c in (ex.get("choices", []) or [])]
            if not choices:
                logging.warning(f"[ex {ex_idx}] No choices in QuALITY, skipping.")
                continue
            q_for_retrieval = quality_prompt(question, choices)
        else:
            q_for_retrieval = question

        # 1) SBERT baseline retrieval
        _, baseline_indices, baseline_ctx = baseline_retrieve_with_indices(
            leaf_embs,
            leaf_texts,
            q_for_retrieval,
            max_tokens=RETRIEVAL_BUDGET,
            top_k=50,
        )

        if not baseline_indices:
            logging.warning(f"[ex {ex_idx}] Baseline retrieved nothing, skipping.")
            continue

        # 2) Post-hoc context = summaries of cluster(s) + baseline leaves
        used_clusters = sorted({int(labels[i]) for i in baseline_indices})
        cluster_ctx_parts = [summaries[cid] for cid in used_clusters if cid in summaries]

        # *** KEY CHANGE: prepend summaries, then leaf chunks ***
        if cluster_ctx_parts:
            posthoc_ctx = "\n\n".join(cluster_ctx_parts + [baseline_ctx])
        else:
            posthoc_ctx = baseline_ctx

        # ------------------------------------------------------------------
        # Dataset-specific QA & scoring
        # ------------------------------------------------------------------

        if dataset == "narrativeqa":
            refs = [_norm_text(r) for r in (ex.get("gold_answers", []) or [])]
            if not refs:
                continue

            # Baseline
            q_trim, c_trim = clip_for_unifiedqa(question, baseline_ctx, budget=UQA_MAX_LEN)
            pred_base = qa.answer_question(c_trim, q_trim)

            # Post-hoc
            q_trim_ph, c_trim_ph = clip_for_unifiedqa(question, posthoc_ctx, budget=UQA_MAX_LEN)
            pred_ph = qa.answer_question(c_trim_ph, q_trim_ph)

            base_bleu1.append(bleu1(refs, pred_base))
            base_bleu4.append(bleu4_equal(refs, pred_base))
            base_rougeL.append(rougeL(refs, pred_base))
            base_meteor.append(meteor_tokenized(refs, pred_base))

            ph_bleu1.append(bleu1(refs, pred_ph))
            ph_bleu4.append(bleu4_equal(refs, pred_ph))
            ph_rougeL.append(rougeL(refs, pred_ph))
            ph_meteor.append(meteor_tokenized(refs, pred_ph))

        elif dataset == "quality":
            choices = [_norm_text(c) for c in (ex.get("choices", []) or [])]
            gold_idx = ex.get("gold_idx", ex.get("correct_idx"))
            if gold_idx is None or not choices:
                continue

            q_with_opts = quality_prompt(question, choices)

            # Baseline
            q_trim, c_trim = clip_for_unifiedqa(q_with_opts, baseline_ctx, budget=UQA_MAX_LEN)
            pred_base = qa.answer_question(c_trim, q_trim)
            guess_base = parse_quality_pred(pred_base, choices)
            base_acc.append(100.0 * float(int(guess_base) == int(gold_idx)))

            # Post-hoc
            q_trim_ph, c_trim_ph = clip_for_unifiedqa(q_with_opts, posthoc_ctx, budget=UQA_MAX_LEN)
            pred_ph = qa.answer_question(c_trim_ph, q_trim_ph)
            guess_ph = parse_quality_pred(pred_ph, choices)
            ph_acc.append(100.0 * float(int(guess_ph) == int(gold_idx)))

        elif dataset == "qasper":
            refs = [_norm_text(r) for r in (ex.get("gold_answers", []) or [])]
            if not refs:
                continue

            # Baseline
            q_trim, c_trim = clip_for_unifiedqa(question, baseline_ctx, budget=UQA_MAX_LEN)
            pred_base = qa.answer_question(c_trim, q_trim)
            base_f1.append(f1_answer(pred_base, refs) * 100.0)

            # Post-hoc
            q_trim_ph, c_trim_ph = clip_for_unifiedqa(question, posthoc_ctx, budget=UQA_MAX_LEN)
            pred_ph = qa.answer_question(c_trim_ph, q_trim_ph)
            ph_f1.append(f1_answer(pred_ph, refs) * 100.0)

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        n_examples_used += 1
        if n_examples_used % 10 == 0:
            logging.info(f"Processed {n_examples_used} examples so far.")

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def mean_safe(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    result: Dict[str, object] = {
        "dataset": dataset,
        "split": str(split_path),
        "n_examples_used": n_examples_used,
    }

    if dataset == "narrativeqa":
        result["baseline"] = {
            "bleu1": mean_safe(base_bleu1),
            "bleu4": mean_safe(base_bleu4),
            "rougeL": mean_safe(base_rougeL),
            "meteor": mean_safe(base_meteor),
        }
        result["posthoc_cluster"] = {
            "bleu1": mean_safe(ph_bleu1),
            "bleu4": mean_safe(ph_bleu4),
            "rougeL": mean_safe(ph_rougeL),
            "meteor": mean_safe(ph_meteor),
        }

    elif dataset == "quality":
        result["baseline"] = {"accuracy": mean_safe(base_acc)}
        result["posthoc_cluster"] = {"accuracy": mean_safe(ph_acc)}

    elif dataset == "qasper":
        result["baseline"] = {"f1": mean_safe(base_f1)}
        result["posthoc_cluster"] = {"f1": mean_safe(ph_f1)}

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logging.info(f"Wrote post-hoc cluster baseline results to {out_path}")
    logging.info(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
