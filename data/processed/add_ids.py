#!/usr/bin/env python3
import json, argparse, hashlib
from pathlib import Path

KEY_BY_DS = {
    "narrativeqa": ("document_id",),
    "quality":     ("article_id",),
    "qasper":      ("paper_id",),   # QASPER already has paper_id
}

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_jsonl(p):
    rows=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def save_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["narrativeqa","quality","qasper"], required=True)
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    id_keys = KEY_BY_DS[args.dataset]
    rows = load_jsonl(args.infile)
    fixed = 0

    for r in rows:
        doc = r.get("doc_text") or r.get("document") or r.get("context") or ""
        # if none of the expected id keys present, add one from hash(doc_text)
        if not any(k in r for k in id_keys):
            if not doc:
                raise SystemExit("Row has no doc_text; cannot derive id.")
            r[id_keys[0]] = sha1(doc)
            fixed += 1

    save_jsonl(args.outfile, rows)
    print(f"Wrote {len(rows)} rows → {args.outfile} (added ids to {fixed} rows).")

if __name__ == "__main__":
    main()
