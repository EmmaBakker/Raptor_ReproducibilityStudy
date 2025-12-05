#!/usr/bin/env python3
"""
Datasets:
- QASPER (allenai/qasper)
- QuALITY (tasksource/QuALITY)
- NarrativeQA manual (deepmind/narrativeqa_manual)

For each dataset under --outdir/<dataset>/ this script writes:
- docs_{train,val,test}.jsonl
- qa_{train,val,test}.jsonl
- docs_titleabs_{train,val,test}.jsonl  (QASPER only)
- corpus.jsonl
- eval_{train,val,test}.jsonl

Usage:
    python prep_datasets.py --datasets all --outdir data/processed
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from datasets import load_dataset
from bs4 import BeautifulSoup 

# Helpers

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)

def dump_jsonl_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def count_jsonl(path: Path) -> int:
    return sum(1 for _ in open(path, "r", encoding="utf-8"))

def clean_docs(df: pd.DataFrame, id_col: str, text_col: str) -> pd.DataFrame:
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df[text_col].str.len() > 0].drop_duplicates(subset=[id_col])
    return df

def make_corpus(dataset_dir: Path, id_col: str, text_col: str) -> Path:
    doc_files = sorted(dataset_dir.glob("docs_*.jsonl"))
    doc_files = [p for p in doc_files if not p.name.startswith("docs_titleabs_")]
    if not doc_files:
        print(f"[WARN] No docs_*.jsonl found in {dataset_dir}")
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
        for kind in ["docs", "qa", "eval"]:
            p = dataset_dir / f"{kind}_{split}.jsonl"
            if p.exists():
                print(f"{split:>5} {kind:5}: {count_jsonl(p)}")

# QASPER

def prep_qasper(base_outdir: Path) -> None:
    print("[*] Loading QASPER …")
    out_dir = base_outdir / "qasper"
    ensure_dir(out_dir)

    def qasper_paper_text(ex: Dict[str, Any]) -> str:
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

    def dedup_keep_order(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for s in seq:
            if s not in seen:
                out.append(s); seen.add(s)
        return out

    def extract_gold_strings_from_answer_obj(a: Any) -> List[str]:
        gold: List[str] = []
        if isinstance(a, dict):
            s = a.get("free_form_answer")
            if isinstance(s, str) and s.strip():
                gold.append(s.strip())
            yn = a.get("yes_no")
            if isinstance(yn, bool):
                gold.append("yes" if yn else "no")
            spans = a.get("extractive_spans")
            if isinstance(spans, list):
                for sp in spans:
                    if isinstance(sp, str) and sp.strip():
                        gold.append(sp.strip())
        elif isinstance(a, str) and a.strip():
            gold.append(a.strip())
        return gold

    def extract_gold_strings_from_answers_field(ans_field: Any) -> List[str]:
        gold: List[str] = []
        if isinstance(ans_field, dict):
            inner = ans_field.get("answer")
            if isinstance(inner, list):
                for a in inner:
                    gold.extend(extract_gold_strings_from_answer_obj(a))
        elif isinstance(ans_field, list):
            for a in ans_field:
                gold.extend(extract_gold_strings_from_answer_obj(a))
        elif isinstance(ans_field, str) and ans_field.strip():
            gold.append(ans_field.strip())
        return dedup_keep_order([g for g in gold if g.strip()])

    try:
        ds = load_dataset("allenai/qasper")
        print("[info] loaded QASPER via datasets.load_dataset")
    except Exception as e:
        raise RuntimeError(f"Cannot load QASPER: {e}")

    def flatten_split(ex_iter) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows_docs, rows_qas = [], []
        for ex in ex_iter:
            pid = ex.get("id")
            title = ex.get("title", "") or ""
            abstract = ex.get("abstract", "") or ""
            full_text = qasper_paper_text(ex)
            rows_docs.append({"paper_id": pid, "title": title, "abstract": abstract, "document": full_text})
            qas = ex.get("qas") or {}
            if isinstance(qas, dict) and isinstance(qas.get("question"), list):
                Q = qas.get("question") or []
                QID = qas.get("question_id") or []
                ANS = qas.get("answers") or []
                for i in range(len(Q)):
                    rows_qas.append({
                        "paper_id": pid,
                        "question_id": QID[i] if i < len(QID) else None,
                        "question": Q[i],
                        "answers": ANS[i] if i < len(ANS) else None,
                    })
            elif isinstance(qas, list):
                for qa in qas:
                    if isinstance(qa, dict):
                        rows_qas.append({
                            "paper_id": pid,
                            "question_id": qa.get("question_id"),
                            "question": qa.get("question"),
                            "answers": qa.get("answers"),
                        })
        docs_df = pd.DataFrame(rows_docs).pipe(clean_docs, id_col="paper_id", text_col="document")
        qas_df = pd.DataFrame(rows_qas)
        return docs_df, qas_df

    docs_tr, qas_tr = flatten_split(ds["train"])
    docs_va, qas_va = flatten_split(ds["validation"])
    docs_te, qas_te = flatten_split(ds["test"])

    save_jsonl(docs_tr, out_dir / "docs_train.jsonl")
    save_jsonl(docs_va, out_dir / "docs_val.jsonl")
    save_jsonl(docs_te, out_dir / "docs_test.jsonl")

    save_jsonl(qas_tr, out_dir / "qa_train.jsonl")
    save_jsonl(qas_va, out_dir / "qa_val.jsonl")
    save_jsonl(qas_te, out_dir / "qa_test.jsonl")

    for split_name, ddf in [("train", docs_tr), ("val", docs_va), ("test", docs_te)]:
        ta = ddf.assign(document=lambda d: (d["title"].fillna("") + "\n\n" + d["abstract"].fillna("")).str.strip())
        save_jsonl(ta[["paper_id", "document"]], out_dir / f"docs_titleabs_{split_name}.jsonl")

    def write_eval(split_name: str, docs_df: pd.DataFrame, qas_df: pd.DataFrame) -> None:
        df = qas_df.merge(docs_df[["paper_id", "document"]], on="paper_id", how="inner")
        rows = []
        for r in df.itertuples(index=False):
            golds = extract_gold_strings_from_answers_field(r.answers)
            if not golds:
                continue
            rows.append({
                "paper_id": r.paper_id,          
                "question_id": getattr(r, "question_id", None), 
                "doc_text": r.document,
                "question": r.question,
                "gold_answers": golds,
            })
        dump_jsonl_rows(rows, out_dir / f"eval_{split_name}.jsonl")


    write_eval("train", docs_tr, qas_tr)
    write_eval("val",   docs_va, qas_va)
    write_eval("test",  docs_te, qas_te)

    make_corpus(out_dir, id_col="paper_id", text_col="document")
    print_counts("qasper", out_dir)

# QuALITY

def _coerce_gold_idx(label_raw, options) -> Optional[int]:
    if label_raw is None:
        return None
    if isinstance(label_raw, (int, float)) or (isinstance(label_raw, str) and label_raw.strip().isdigit()):
        i = int(label_raw)
        if options and 1 <= i <= len(options):
            return i - 1
        if 0 <= i < (len(options) if options else max(i+1, 4)):
            return i
        return None
    if isinstance(label_raw, str):
        s = label_raw.strip()
        if len(s) == 1 and "A" <= s.upper() <= "Z":
            return ord(s.upper()) - ord("A")
        if options:
            try:
                return options.index(s)
            except ValueError:
                return None
    return None

def _coerce_is_hard(v) -> Optional[bool]:
    if v is None: return None
    try:
        return bool(int(v))
    except Exception:
        return bool(v)

def prep_quality(base_outdir: Path) -> None:
    print("[*] Loading QuALITY (tasksource/QuALITY)…")
    ds = load_dataset("tasksource/QuALITY", trust_remote_code=True)
    out_dir = base_outdir / "quality"
    ensure_dir(out_dir)

    def flatten_split(split) -> Tuple[pd.DataFrame, pd.DataFrame]:
        docs, qas = [], []
        for ex in split:
            article_id = ex.get("article_id") or ex.get("id") or ex.get("doc_id")
            article_txt = ex.get("article") or ex.get("context") or ex.get("passage") or ""
            docs.append({"article_id": article_id, "document": article_txt})
            q_text  = ex.get("question") or ex.get("prompt") or ""
            options = ex.get("options")  or ex.get("choices") or []
            qid     = ex.get("qid")      or ex.get("question_id")
            label_raw = ex.get("label", ex.get("gold_label", ex.get("gold_idx", None)))
            gold_idx = _coerce_gold_idx(label_raw, options)
            is_hard  = _coerce_is_hard(ex.get("difficult", ex.get("is_hard", None)))
            qas.append({
                "article_id": article_id,
                "qid": qid,
                "question": q_text,
                "options": options,
                "gold_label": gold_idx,
                "is_hard": is_hard,
            })
        return (
            pd.DataFrame(docs).pipe(clean_docs, id_col="article_id", text_col="document"),
            pd.DataFrame(qas),
        )

    split_plan = []
    if "train" in ds: split_plan.append(("train", "train"))
    if "validation" in ds: split_plan.append(("val", "validation"))
    if "test" in ds: split_plan.append(("test", "test"))

    for out_name, hf_name in split_plan:
        docs_df, qas_df = flatten_split(ds[hf_name])
        save_jsonl(docs_df, out_dir / f"docs_{out_name}.jsonl")
        save_jsonl(qas_df,  out_dir / f"qa_{out_name}.jsonl")

        df = qas_df.merge(docs_df, on="article_id", how="inner")
        rows = []
        for r in df.itertuples(index=False):
            rows.append({
                "doc_text": r.document,
                "question": r.question,
                "choices": list(r.options),
                "gold_idx": int(r.gold_label) if r.gold_label is not None else None,
                "is_hard": bool(r.is_hard) if r.is_hard is not None else None,
            })
        dump_jsonl_rows(rows, out_dir / f"eval_{out_name}.jsonl")

    make_corpus(out_dir, id_col="article_id", text_col="document")
    print_counts("quality", out_dir)

# NarrativeQA

def get_narrativeqa_dir() -> Path:
    p = Path("data/raw/narrativeqa_stories/tmp_nqa/tmp")
    if not p.exists():
        raise RuntimeError("NarrativeQA stories not found at data/raw/narrativeqa_stories/tmp_nqa/tmp")
    return p

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def prep_narrativeqa(base_outdir: Path) -> None:
    print("[*] Loading NarrativeQA manual (deepmind/narrativeqa_manual)…")
    data_dir = get_narrativeqa_dir()
    ds = load_dataset("deepmind/narrativeqa_manual", data_dir=str(data_dir))
    out_dir = base_outdir / "narrativeqa"
    ensure_dir(out_dir)

    def get_doc_text(r: Dict[str, Any]) -> str:
        doc = r.get("document")
        text = ""
        if isinstance(doc, str):
            text = doc
        elif isinstance(doc, dict):
            text = doc.get("text") or doc.get("story_text") or doc.get("raw") or ""
        else:
            for k in ("story", "text"):
                if isinstance(r.get(k), str):
                    text = r[k]
                    break
        text = str(text).strip()
        # If it looks like HTML (starts with <html> or has <body> tags), strip it.
        if text.startswith("<") and "</" in text:
            try:
                text = BeautifulSoup(text, "html.parser").get_text(separator="\n")
            except Exception:
                pass
        return text


    def get_question_text(r: Dict[str, Any]) -> str:
        q = r.get("question")
        if isinstance(q, str): return q.strip()
        if isinstance(q, dict):
            return str(q.get("text") or q.get("q") or "").strip()
        return ""

    def normalize_answers(ans: Any) -> List[str]:
        out: List[str] = []
        if isinstance(ans, list):
            for a in ans:
                if isinstance(a, str) and a.strip():
                    out.append(a.strip())
                elif isinstance(a, dict):
                    t = a.get("text") or a.get("answer") or a.get("a")
                    if isinstance(t, str) and t.strip():
                        out.append(t.strip())
        elif isinstance(ans, dict):
            t = ans.get("text") or ans.get("answer") or ans.get("a")
            if isinstance(t, str) and t.strip():
                out.append(t.strip())
        elif isinstance(ans, str) and ans.strip():
            out.append(ans.strip())
        seen, uniq = set(), []
        for s in out:
            if s not in seen:
                uniq.append(s); seen.add(s)
        return uniq

    available = [s for s in ["train", "validation", "test"] if s in ds]
    for split in available:
        rows = [dict(r) for r in ds[split]]
        doc_rows, qa_rows = [], []
        for r in rows:
            doc_text = get_doc_text(r)
            if not doc_text: continue
            doc_id = _sha1(doc_text)
            doc_rows.append({"document_id": doc_id, "title": "", "document": doc_text})
            q_text = get_question_text(r)
            if q_text:
                golds = normalize_answers(r.get("answers"))
                qa_rows.append({
                    "document_id": doc_id,
                    "question_id": _sha1(doc_id + q_text),
                    "question": q_text,
                    "answers": golds,
                })
        docs_df = pd.DataFrame(doc_rows).drop_duplicates(subset=["document_id"]).pipe(clean_docs, id_col="document_id", text_col="document")
        qas_df = pd.DataFrame(qa_rows)
        out_name = {"train": "train", "validation": "val", "test": "test"}[split]
        save_jsonl(docs_df[["document_id", "title", "document"]], out_dir / f"docs_{out_name}.jsonl")
        save_jsonl(qas_df, out_dir / f"qa_{out_name}.jsonl")

        merged = qas_df.merge(docs_df[["document_id", "document"]], on="document_id", how="inner")
        eval_rows = []
        for r in merged.itertuples(index=False):
            golds = [g for g in (r.answers or []) if isinstance(g, str) and g.strip()]
            if not golds:
                continue
            eval_rows.append({
                "doc_text": r.document,
                "question": r.question,
                "gold_answers": golds
            })
        dump_jsonl_rows(eval_rows, out_dir / f"eval_{out_name}.jsonl")

    make_corpus(out_dir, id_col="document_id", text_col="document")
    print_counts("narrativeqa", out_dir)

# CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all","qasper","quality","narrativeqa"])
    parser.add_argument("--outdir", default="data/processed")
    args = parser.parse_args()

    base_outdir = Path(args.outdir)
    ensure_dir(base_outdir)

    todo = set(args.datasets)
    if "all" in todo:
        todo = {"qasper","quality","narrativeqa"}

    if "qasper" in todo:
        prep_qasper(base_outdir)
    if "quality" in todo:
        prep_quality(base_outdir)
    if "narrativeqa" in todo:
        prep_narrativeqa(base_outdir)

    print("\n[DONE] Dataset preparation complete.")

if __name__ == "__main__":
    main()