#!/usr/bin/env python3
"""
Datasets:
- QASPER (allenai/qasper)
- QuALITY (tasksource/QuALITY)
- NarrativeQA manual (deepmind/narrativeqa_manual)  -> requires local stories

Outputs (per dataset under --outdir/<dataset>/):
- docs_{train,val,test}.jsonl
- qa_{train,val,test}.jsonl
- docs_titleabs_{train,val,test}.jsonl  (QASPER only)
- corpus.jsonl                          (merged unique docs with standardized fields: doc_id, text)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
from datasets import load_dataset
import hashlib

# -----------------------
# Helpers
# -----------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)

def count_jsonl(path: Path) -> int:
    return sum(1 for _ in open(path, "r", encoding="utf-8"))

def make_corpus(dataset_dir: Path, id_col: str, text_col: str) -> Path:
    """Merge all docs_*.jsonl (excluding title+abstract files) into corpus.jsonl (doc_id, text)."""
    doc_files = sorted(dataset_dir.glob("docs_*.jsonl"))
    doc_files = [p for p in doc_files if not p.name.startswith("docs_titleabs_")]
    if not doc_files:
        print(f"[WARN] No docs_*.jsonl (excluding titleabs) found in {dataset_dir}")
        return dataset_dir / "corpus.jsonl"

    dfs = [pd.read_json(str(f), lines=True) for f in doc_files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=[id_col])

    out_df = df[[id_col, text_col]].rename(columns={id_col: "doc_id", text_col: "text"})
    out_path = dataset_dir / "corpus.jsonl"
    save_jsonl(out_df, out_path)
    print(f"[OK] Wrote corpus: {out_path} ({len(out_df)} docs)")
    return out_path

def print_counts(dataset_name: str, dataset_dir: Path) -> None:
    print(f"\n== {dataset_name.upper()} ==")
    for split in ["train", "val", "test"]:
        d = dataset_dir / f"docs_{split}.jsonl"
        q = dataset_dir / f"qa_{split}.jsonl"
        if d.exists():
            print(f"{split:>5} docs : {count_jsonl(d)}")
        if q.exists():
            print(f"{split:>5} qas  : {count_jsonl(q)}")

def clean_docs(df: pd.DataFrame, id_col: str, text_col: str) -> pd.DataFrame:
    """Drop empty texts and de-duplicate by id_col."""
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df[text_col].str.len() > 0].drop_duplicates(subset=[id_col])
    return df

# -----------------------
# QASPER
# -----------------------
def qasper_paper_text(ex: Dict[str, Any]) -> str:
    """
    Build a single string from ex['full_text'] (dict with 'section_name': List[str], 'paragraphs': List[List[str]]).
    """
    ft = ex.get("full_text") or {}
    sec_names = ft.get("section_name", []) or []
    paras = ft.get("paragraphs", []) or []

    blocks = []
    for i in range(max(len(sec_names), len(paras))):
        name = sec_names[i] if i < len(sec_names) and sec_names[i] else ""
        if name:
            blocks.append(name)
        if i < len(paras) and paras[i]:
            blocks.extend([p for p in paras[i] if p])
    return "\n\n".join(blocks).strip()

def prep_qasper(base_outdir: Path) -> None:
    print("[*] Loading QASPER (allenai/qasper)...")
    ds = load_dataset("allenai/qasper", trust_remote_code=True)

    out_dir = base_outdir / "qasper"
    ensure_dir(out_dir)

    def flatten_split(split) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows_docs, rows_qas = [], []
        for ex in split:
            pid = ex["id"]  # HF uses 'id'
            title = ex.get("title", "") or ""
            abstract = ex.get("abstract", "") or ""
            full_text = qasper_paper_text(ex)

            rows_docs.append({
                "paper_id": pid,
                "title": title,
                "abstract": abstract,
                "document": full_text
            })

            qas = ex.get("qas", {}) or {}
            questions = qas.get("question", []) or []
            qids = qas.get("question_id", []) or []
            answers_list = qas.get("answers", []) or []  # may be list[str] or list[dict]

            for i in range(len(questions)):
                q_text = questions[i]
                q_id = qids[i] if i < len(qids) else None
                raw_ans = answers_list[i] if i < len(answers_list) else None

                # normalize answers (string or dict) to list of dicts
                norm_answers = []
                if isinstance(raw_ans, list):
                    for a in raw_ans:
                        if isinstance(a, dict):
                            norm_answers.append(a)
                        elif isinstance(a, str):
                            norm_answers.append({"free_form_answer": a})
                elif isinstance(raw_ans, dict):
                    norm_answers.append(raw_ans)
                elif isinstance(raw_ans, str):
                    norm_answers.append({"free_form_answer": raw_ans})

                rows_qas.append({
                    "paper_id": pid,
                    "question_id": q_id,
                    "question": q_text,
                    "answers": norm_answers,
                })

        return (
            pd.DataFrame(rows_docs).pipe(clean_docs, id_col="paper_id", text_col="document"),
            pd.DataFrame(rows_qas)
        )

    docs_tr, qas_tr = flatten_split(ds["train"])
    docs_va, qas_va = flatten_split(ds["validation"])
    docs_te, qas_te = flatten_split(ds["test"])

    save_jsonl(docs_tr, out_dir / "docs_train.jsonl")
    save_jsonl(docs_va, out_dir / "docs_val.jsonl")
    save_jsonl(docs_te, out_dir / "docs_test.jsonl")

    save_jsonl(qas_tr, out_dir / "qa_train.jsonl")
    save_jsonl(qas_va, out_dir / "qa_val.jsonl")
    save_jsonl(qas_te, out_dir / "qa_test.jsonl")

    # Title + Abstract baseline (for ablations)
    for split_name, ddf in [("train", docs_tr), ("val", docs_va), ("test", docs_te)]:
        ta = ddf.assign(document=lambda d: (d["title"].fillna("") + "\n\n" + d["abstract"].fillna("")).str.strip())
        save_jsonl(ta[["paper_id", "document"]], out_dir / f"docs_titleabs_{split_name}.jsonl")

    make_corpus(out_dir, id_col="paper_id", text_col="document")
    print_counts("qasper", out_dir)

# -----------------------
# QuALITY
# -----------------------
def prep_quality(base_outdir: Path) -> None:
    print("[*] Loading QuALITY (tasksource/QuALITY)...")
    ds = load_dataset("tasksource/QuALITY", trust_remote_code=True)

    out_dir = base_outdir / "quality"
    ensure_dir(out_dir)

    available = set(ds.keys())  # e.g., {'train','validation'} or also 'test'
    print(f"[info] QuALITY available splits: {sorted(available)}")

    def flatten_split(split) -> Tuple[pd.DataFrame, pd.DataFrame]:
        docs, qas = [], []
        for ex in split:
            article_id = ex.get("article_id") or ex.get("id") or ex.get("doc_id")
            article_txt = ex.get("article") or ex.get("context") or ex.get("passage") or ""
            docs.append({"article_id": article_id, "document": article_txt})

            q_text = ex.get("question") or ex.get("prompt") or ""
            options = ex.get("options") or ex.get("choices") or []
            qid = ex.get("qid") or ex.get("question_id")

            label_raw = (
                ex.get("label", None) if "label" in ex else
                ex.get("gold_label", None) if "gold_label" in ex else
                ex.get("gold_idx", None) if "gold_idx" in ex else
                None
            )

            gold_idx = None
            if isinstance(label_raw, int):
                gold_idx = label_raw
            elif isinstance(label_raw, float):
                gold_idx = int(label_raw)
            elif isinstance(label_raw, str):
                s = label_raw.strip().upper()
                if len(s) == 1 and "A" <= s <= "Z":
                    gold_idx = ord(s) - ord("A")
                else:
                    try:
                        gold_idx = options.index(label_raw)
                    except Exception:
                        gold_idx = None

            qas.append({
                "article_id": article_id,
                "qid": qid,
                "question": q_text,
                "options": options,
                "gold_label": gold_idx,
                "is_hard": ex.get("is_hard", None)
            })

        return (
            pd.DataFrame(docs).pipe(clean_docs, id_col="article_id", text_col="document"),
            pd.DataFrame(qas),
        )

    split_plan = []
    if "train" in available:
        split_plan.append(("train", "train"))
    if "validation" in available:
        split_plan.append(("val", "validation"))
    if "test" in available:
        split_plan.append(("test", "test"))

    for out_name, hf_name in split_plan:
        docs_df, qas_df = flatten_split(ds[hf_name])
        save_jsonl(docs_df, out_dir / f"docs_{out_name}.jsonl")
        save_jsonl(qas_df,  out_dir / f"qa_{out_name}.jsonl")

    make_corpus(out_dir, id_col="article_id", text_col="document")
    print_counts("quality", out_dir)

# -----------------------
# NarrativeQA (manual)
# -----------------------
"""
To download NarrativeQA stories:
    mkdir -p data/raw/narrativeqa_stories
    cd data/raw/narrativeqa_stories
    git clone https://github.com/deepmind/narrativeqa tmp_nqa
    bash tmp_nqa/download_stories.sh
This script expects the files under: data/raw/narrativeqa_stories/tmp_nqa/tmp
"""

def get_narrativeqa_dir() -> Path:
    p = Path("data/raw/narrativeqa_stories/tmp_nqa/tmp")
    if not p.exists() or not p.is_dir():
        raise RuntimeError(
            "NarrativeQA stories not found at data/raw/narrativeqa_stories/tmp_nqa/tmp.\n"
            "Run download_stories.sh or adjust this path."
        )
    return p


def prep_narrativeqa(base_outdir: Path) -> None:
    print("[*] Loading NarrativeQA manual (deepmind/narrativeqa_manual)...")
    data_dir = get_narrativeqa_dir()
    ds = load_dataset("deepmind/narrativeqa_manual", data_dir=str(data_dir), trust_remote_code=True)

    out_dir = base_outdir / "narrativeqa"
    ensure_dir(out_dir)

    available = [s for s in ["train", "validation", "test"] if s in ds]
    if not available:
        raise RuntimeError("No splits found in narrativeqa_manual dataset.")

    def sha1(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    for split in available:
        df = pd.DataFrame(ds[split])
        cols = set(df.columns)

        # The split you have: ['answers', 'document', 'question']
        # So we take 'document' as the text column and synthesize document_id by hashing it.
        if "document" not in cols:
            raise RuntimeError(
                f"NarrativeQA: couldn't find a text column in split '{split}'. "
                f"Columns present: {sorted(list(cols))}"
            )

        text_col = "document"
        title_col = None  # not available
        question_col = "question" if "question" in cols else None
        answers_col  = "answers"  if "answers"  in cols else None

        # ---- Build docs_{split}.jsonl
        df["_document_id"] = df[text_col].astype(str).map(sha1)
        docs_df = (
            df[["_document_id", text_col]]
            .drop_duplicates(subset=["_document_id"])
            .rename(columns={"_document_id": "document_id", text_col: "document"})
        )
        docs_df["title"] = ""  # no titles in this layout
        docs_df = clean_docs(docs_df, id_col="document_id", text_col="document")

        # ---- Build qa_{split}.jsonl (only rows that have a question)
        if question_col:
            qa_rows = []
            for r in df.itertuples(index=False):
                rdict = r._asdict()
                doc_text = str(rdict[text_col])
                doc_id = sha1(doc_text)

                q_text = rdict.get(question_col, None)
                if not q_text or not str(q_text).strip():
                    continue

                # Normalize answers to a list[str]
                ans = rdict.get(answers_col, None)
                answers_norm = []
                if isinstance(ans, list):
                    answers_norm = [a for a in ans if isinstance(a, str) and a.strip()]
                elif isinstance(ans, str):
                    answers_norm = [ans] if ans.strip() else []

                # Provide a deterministic question_id (hash of doc_id + question)
                qid = sha1(doc_id + "\n" + str(q_text))

                qa_rows.append({
                    "document_id": doc_id,
                    "question_id": qid,
                    "question": q_text,
                    "answers": answers_norm,
                })
            qas_df = pd.DataFrame(qa_rows)
        else:
            qas_df = pd.DataFrame(columns=["document_id", "question_id", "question", "answers"])

        out_name = {"train": "train", "validation": "val", "test": "test"}[split]
        save_jsonl(docs_df[["document_id", "title", "document"]], out_dir / f"docs_{out_name}.jsonl")
        save_jsonl(qas_df, out_dir / f"qa_{out_name}.jsonl")

    make_corpus(out_dir, id_col="document_id", text_col="document")
    print_counts("narrativeqa", out_dir)


# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "qasper", "quality", "narrativeqa"],
        help="Which datasets to prepare."
    )
    parser.add_argument(
        "--outdir",
        default="data/processed",
        help="Base output directory."
    )
    args = parser.parse_args()

    base_outdir = Path(args.outdir)
    ensure_dir(base_outdir)

    todo = set(args.datasets)
    if "all" in todo:
        todo = {"qasper", "quality", "narrativeqa"}

    if "qasper" in todo:
        prep_qasper(base_outdir)
    if "quality" in todo:
        prep_quality(base_outdir)
    if "narrativeqa" in todo:
        prep_narrativeqa(base_outdir)

    print("\n[DONE] Dataset preparation complete.")

if __name__ == "__main__":
    main()
