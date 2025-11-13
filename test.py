#!/usr/bin/env python3
import pickle, os, statistics, argparse
from pathlib import Path
import tiktoken

TOK = tiktoken.get_encoding("cl100k_base")

def token_len(s: str) -> int:
    return len(TOK.encode(s or ""))

def per_dataset_stats(tree_dir: Path):
    parent_summary_lens = []
    child_lens_all = []      # every child occurrence (each edge) or unique? see note below
    children_counts = []
    comp_ratios = []

    pkl_files = list(tree_dir.glob("**/*.pkl"))
    for pkl in pkl_files:
        with open(pkl, "rb") as f:
            t = pickle.load(f)

        # Gather once for efficient lookups
        all_nodes = t.all_nodes                      # dict: idx -> Node
        # parent = any node that has children
        for node in all_nodes.values():
            if not node.children:
                continue
            # parent stats
            ps = token_len(node.text)
            parent_summary_lens.append(ps)
            kids = [all_nodes[i] for i in node.children]
            kc   = len(kids)
            children_counts.append(kc)
            kid_lens = [token_len(k.text) for k in kids]
            child_lens_all.extend(kid_lens)
            denom = sum(kid_lens) if kid_lens else 1
            comp_ratios.append(ps / denom)

    if not parent_summary_lens:
        return None

    return {
        "avg_summary_len_tokens": statistics.mean(parent_summary_lens),
        "avg_child_len_tokens": statistics.mean(child_lens_all) if child_lens_all else 0.0,
        "avg_children_per_parent": statistics.mean(children_counts) if children_counts else 0.0,
        "avg_compression_ratio": statistics.mean(comp_ratios) if comp_ratios else 0.0,
        "num_parents": len(parent_summary_lens),
        "num_pkls": len(pkl_files),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/raptor_trees", help="root dir with dataset subfolders")
    ap.add_argument("--datasets", nargs="+", default=["qasper","quality", "narrativeqa"], help="e.g. qasper quality narrativeqa")
    args = ap.parse_args()

    for ds in args.datasets:
        # collect across all seeds under this dataset
        ds_dir = Path(args.base) / ds
        if not ds_dir.exists():
            print(f"[warn] no dir: {ds_dir}")
            continue
        stats = per_dataset_stats(ds_dir)
        if not stats:
            print(f"[{ds}] no parents found")
            continue
        print(f"\n== {ds.upper()} ==")
        print(f"Avg. Summary Length (tokens): {stats['avg_summary_len_tokens']:.1f}")
        print(f"Avg. Child Node Text Length (tokens): {stats['avg_child_len_tokens']:.1f}")
        print(f"Avg. # of Child Nodes Per Parent: {stats['avg_children_per_parent']:.1f}")
        print(f"Avg. Compression Ratio: {stats['avg_compression_ratio']:.2f}")
        print(f"(parents={stats['num_parents']}, pkls={stats['num_pkls']})")

if __name__ == "__main__":
    main()
