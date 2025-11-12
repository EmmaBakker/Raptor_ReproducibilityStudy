#!/usr/bin/env python3
import json, argparse, random, statistics
from pathlib import Path
import tiktoken
TOK = tiktoken.get_encoding("cl100k_base")

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def norm_text(x):
    if isinstance(x, dict):
        for k in ("doc_text","document","doc","text"):
            if k in x and isinstance(x[k], str): return x[k]
        # fallback to any str value
        for v in x.values():
            if isinstance(v, str): return v
        return str(x)
    return str(x)

def doc_key(row, dataset):
    if dataset=="narrativeqa":
        return row.get("document_id")
    if dataset=="quality":
        return row.get("article_id")
    if dataset=="qasper":
        return row.get("paper_id")
    # fallback: hash of doc text
    import hashlib
    return hashlib.sha1(norm_text(row).encode("utf-8")).hexdigest()

def token_len(s): return len(TOK.encode(s or ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["narrativeqa","quality","qasper"], required=True)
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--num_docs", type=int, default=50)
    ap.add_argument("--max_q_per_doc", type=int, default=0, help="0 = keep all questions")
    ap.add_argument("--seed", type=int, default=224)
    args = ap.parse_args()
    random.seed(args.seed)

    # group rows by doc
    rows = list(load_jsonl(args.infile))
    by_doc = {}
    doc_texts = {}
    for r in rows:
        k = doc_key(r, args.dataset)
        by_doc.setdefault(k, []).append(r)
        if k not in doc_texts:
            doc_texts[k] = norm_text(r.get("doc_text") or r.get("document") or r.get("doc") or "")

    # compute doc lengths
    doc_lengths = {k: token_len(doc_texts[k]) for k in by_doc}

    # stratify by quartiles
    docs = list(by_doc.keys())
    lens = [doc_lengths[k] for k in docs]
    if len(docs) <= args.num_docs:
        selected_docs = docs
    else:
        q1,q2,q3 = statistics.quantiles(lens, n=4, method="inclusive")
        bins = {0:[],1:[],2:[],3:[]}
        for k in docs:
            L = doc_lengths[k]
            b = 0 if L<=q1 else 1 if L<=q2 else 2 if L<=q3 else 3
            bins[b].append(k)
        # even allocation across bins
        target = [args.num_docs//4]*4
        for i in range(args.num_docs % 4):
            target[-(i+1)] += 1  # give remainder to longer bins
        selected_docs=[]
        for b in range(4):
            pool = bins[b]
            random.shuffle(pool)
            take = min(target[b], len(pool))
            selected_docs.extend(pool[:take])

    # assemble subset rows
    out_rows = []
    for k in selected_docs:
        qs = by_doc[k]
        if args.max_q_per_doc and len(qs) > args.max_q_per_doc:
            random.Random(args.seed + hash(k) % (10**6)).shuffle(qs)
            qs = qs[:args.max_q_per_doc]
        out_rows.extend(qs)

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Selected {len(selected_docs)} docs, {len(out_rows)} Qs.")
    # rough distribution report
    ls = [doc_lengths[k] for k in selected_docs]
    print(f"Doc length stats (tokens): min={min(ls)}, median={int(statistics.median(ls))}, max={max(ls)}")

if __name__ == "__main__":
    main()
