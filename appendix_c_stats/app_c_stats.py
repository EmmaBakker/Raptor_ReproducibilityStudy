#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import tiktoken
from tqdm import tqdm

# RAPTOR repo imports (installed in editable mode)
from raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.SummarizationModels import GPT3TurboSummarizationModel
from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor.utils import get_text
from raptor.tree_structures import Tree, Node

import os, random, numpy as np

def set_global_seed(seed: int = 224):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

set_global_seed(224)

# Build a RAPTOR tree
def build_tree(text: str,
               chunk_tokens: int = 100,
               summarization_len: int = 170,
               num_layers_cap: int = 8) -> Tree:
    cfg = RetrievalAugmentationConfig(
        # TreeBuilder (chunking, clustering, summarization)
        tb_max_tokens=chunk_tokens,
        tb_num_layers=num_layers_cap,
        tb_threshold=0.10,
        tb_summarization_length=summarization_len,
        summarization_model=GPT3TurboSummarizationModel("gpt-3.5-turbo"),
        # Embeddings (SBERT for ALL nodes)
        embedding_model=SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1"),
        tb_cluster_embedding_model="EMB",
        # Retriever (keep consistent key)
        tr_context_embedding_model="EMB",
    )
    ra = RetrievalAugmentation(cfg)
    tree = ra.tree_builder.build_from_text(text, use_multithreading=True)
    return tree


# Per-doc, per-layer rows (Appendix C style)
def appendix_c_rows(tree: Tree, tokenizer) -> List[Dict]:
    """
    One row per (doc, layer) with:
      nodes = #parents at this layer
      avg_children = average #children per parent
      avg_child_len_sum = TOTAL child tokens across parents in this layer
      avg_child_len_per_child = total child tokens / total #children
      avg_summary_len = average parent-summary tokens
      compression_ratio = sum(summary tokens) / sum(child tokens)   (pooled)
    """
    rows = []
    for L in sorted(tree.layer_to_nodes.keys()):
        if L == 0:
            continue
        parents: List[Node] = tree.layer_to_nodes[L]
        if not parents:
            continue

        per_parent_children_ct = []
        per_parent_child_len_sum = []
        per_parent_summary_len = []

        for p in parents:
            if not p.children:
                continue
            kids = [tree.all_nodes[i] for i in sorted(p.children)]
            child_text = get_text(kids)

            child_len_sum = len(tokenizer.encode(child_text))
            summary_len = len(tokenizer.encode(p.text))
            n_children = len(kids)

            per_parent_children_ct.append(n_children)
            per_parent_child_len_sum.append(child_len_sum)
            per_parent_summary_len.append(summary_len)

        if not per_parent_children_ct:
            continue

        total_children = sum(per_parent_children_ct)
        total_child_len_sum = sum(per_parent_child_len_sum)
        total_summary_len = sum(per_parent_summary_len)

        avg_children = total_children / len(per_parent_children_ct)
        avg_child_len_per_child = (total_child_len_sum / total_children) if total_children > 0 else 0.0
        avg_summary_len = total_summary_len / len(per_parent_summary_len)
        compression = (total_summary_len / total_child_len_sum) if total_child_len_sum > 0 else 0.0

        rows.append(dict(
            layer=L,
            nodes=len(parents),
            avg_children=avg_children,
            # NOTE: TOTAL, kept to match CSV header name
            avg_child_len_sum=total_child_len_sum,
            avg_child_len_per_child=avg_child_len_per_child,
            avg_summary_len=avg_summary_len,
            compression_ratio=compression,
        ))
    return rows


# Weighted (pooled) layer/ALL summaries
def summarize_weighted(rows: List[Dict], layer_key=None) -> Dict:
    agg = {
        "parents": 0,
        "sum_children": 0.0,      # total #children
        "sum_child_len_sum": 0.0,  # total child tokens
        "sum_summary_len": 0.0,    # total summary tokens
    }
    for r in rows:
        n = int(r["nodes"])  # #parents in this (doc, layer) row
        agg["parents"] += n
        agg["sum_children"] += float(r["avg_children"]) * n
        # avg_child_len_sum is already a TOTAL for the row → do NOT multiply by n
        agg["sum_child_len_sum"] += float(r["avg_child_len_sum"])
        # avg_summary_len is per-parent → multiply by n to get totals
        agg["sum_summary_len"] += float(r["avg_summary_len"]) * n

    if agg["parents"] == 0:
        return {
            "layer": layer_key if layer_key is not None else "ALL",
            "parents": 0,
            "avg_children": 0.0,
            "avg_child_len_per_child": 0.0,
            "avg_summary_len": 0.0,
            "compression_ratio": 0.0,
        }

    total_children = agg["sum_children"]
    avg_children = agg["sum_children"] / agg["parents"]
    avg_child_len_per_child = (agg["sum_child_len_sum"] / total_children) if total_children > 0 else 0.0
    avg_summary_len = agg["sum_summary_len"] / agg["parents"]
    compression = (agg["sum_summary_len"] / agg["sum_child_len_sum"]) if agg["sum_child_len_sum"] > 0 else 0.0

    return {
        "layer": layer_key if layer_key is not None else "ALL",
        "parents": agg["parents"],
        "avg_children": avg_children,
        "avg_child_len_per_child": avg_child_len_per_child,
        "avg_summary_len": avg_summary_len,
        "compression_ratio": compression,
    }


# Main runner
def run(dataset: str, num_docs: int, base_dir: Path):
    corpus = base_dir / dataset / "corpus.jsonl"
    assert corpus.exists(), f"Missing {corpus}. Did you run prep_datasets.py?"

    out_dir = Path("appendix_c_stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"stats_{dataset}.csv"
    out_csv_summary = out_dir / f"stats_{dataset}_summary.csv"

    tok = tiktoken.get_encoding("cl100k_base")
    rows_out: List[Dict] = []

    with open(corpus, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=num_docs, desc=f"{dataset}: building + scoring", unit="doc")):
            if i >= num_docs:
                break
            rec = json.loads(line)
            doc_id = rec["doc_id"]
            text = rec["text"]
            if not text or not isinstance(text, str):
                continue

            try:
                tree = build_tree(text)
                rows = appendix_c_rows(tree, tokenizer=tok)
                for r in rows:
                    r_out = {"doc_id": doc_id, **r}
                    rows_out.append(r_out)
            except Exception as e:
                print(f"[WARN] Skipping doc_id={doc_id}: {e}")

    # Write detailed per-doc×layer CSV
    fieldnames = [
        "doc_id", "layer", "nodes", "avg_children",
        "avg_child_len_sum", "avg_child_len_per_child",
        "avg_summary_len", "compression_ratio"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as w:
        writer = csv.DictWriter(w, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # Group by layer and compute pooled layer-wise + overall summaries
    by_layer = defaultdict(list)
    for r in rows_out:
        by_layer[r["layer"]].append(r)

    summary_rows = []
    for L in sorted(by_layer.keys()):
        summary_rows.append(summarize_weighted(by_layer[L], layer_key=L))
    summary_rows.append(summarize_weighted(rows_out, layer_key="ALL"))

    with open(out_csv_summary, "w", newline="", encoding="utf-8") as w:
        writer = csv.DictWriter(
            w,
            fieldnames=[
                "layer", "parents", "avg_children",
                "avg_child_len_per_child",
                "avg_summary_len", "compression_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] Wrote detailed per-layer file → {out_csv}")
    print(f"[OK] Wrote Appendix C summary → {out_csv_summary}")


# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["qasper", "quality", "narrativeqa"], required=True)
    p.add_argument("--num_docs", type=int, default=30, help="Number of documents to process.")
    p.add_argument("--base_dir", default="data/processed", help="Base dir containing <dataset>/corpus.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(dataset=args.dataset, num_docs=args.num_docs, base_dir=Path(args.base_dir))
