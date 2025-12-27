#!/usr/bin/env python3
"""
summarize_diagnostics.py

Create compact, paper-friendly summaries from one or more diagnostics JSON files.

Typical workflow for your folder structure:
  python analysis/summarize_diagnostics.py \
    --scan-dirs results_raptor results_trimap results_pacmap results_hdbscan

This will look for *.json files starting with "diagnostics" in those folders and
write a compact summary next to each input file:
  <dir>/diagnostics_x.json -> <dir>/summary_diagnostics_x.json

You can also run it on a single file:
  python analysis/summarize_diagnostics.py --in results_raptor/diagnostics_umap_raptor.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import math


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def mean_std(vals: List[float]) -> Dict[str, Any]:
    clean = [float(v) for v in vals if _is_num(v)]
    if not clean:
        return {"mean": None, "std": None, "n_seeds": 0}
    m = sum(clean) / len(clean)
    var = sum((v - m) ** 2 for v in clean) / len(clean)
    return {"mean": m, "std": math.sqrt(var), "n_seeds": len(clean)}


def _get_seed_block(per_seed: Dict[str, Any], seed: int) -> Optional[Dict[str, Any]]:
    # JSON may have string keys ("224") or int keys (224) depending on who wrote it
    if str(seed) in per_seed:
        return per_seed[str(seed)]
    if seed in per_seed:
        return per_seed[seed]
    return None


def _pull_agg(dset_agg: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = dset_agg.get(key, {})
    return v if isinstance(v, dict) else {}


def _metric_from_agg(dset_agg: Dict[str, Any], agg_key: str) -> Optional[Dict[str, Any]]:
    v = _pull_agg(dset_agg, agg_key)
    if not v:
        return None
    return {"mean": v.get("mean"), "std": v.get("std"), "n_seeds": v.get("n_seeds")}


def _metric_from_per_seed(per_seed: Dict[str, Any], seeds: List[int], path: List[str]) -> Optional[Dict[str, Any]]:
    """
    Extract a scalar metric from each seed result and compute mean/std over seeds.
    path example:
      ["retrieval_stats", "baseline_ctx_tokens", "mean"]
    """
    vals: List[float] = []
    for s in seeds:
        blk = _get_seed_block(per_seed, s)
        if not blk:
            continue
        cur: Any = blk
        ok = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok and _is_num(cur):
            vals.append(float(cur))
    if not vals:
        return None
    return mean_std(vals)


def summarize_one(inp_path: Path, out_path: Optional[Path] = None) -> Path:
    with inp_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    datasets = raw.get("datasets", {}) or {}
    seeds = raw.get("seeds", []) or []
    seeds = [int(s) for s in seeds if isinstance(s, (int, str)) and str(s).isdigit()]

    compact: Dict[str, Any] = {
        "input": str(inp_path),
        "trees_root": raw.get("trees_root"),
        "tree_subdir_template": raw.get("tree_subdir_template"),
        "seeds": seeds,
        "max_q": raw.get("max_q"),
        "do_overlap": raw.get("do_overlap"),
        "full_tree_metrics": raw.get("full_tree_metrics"),
        "datasets": {},
    }

    for dname, d in datasets.items():
        agg = (d.get("aggregate_over_seeds", {}) or {})
        per_seed = (d.get("per_seed", {}) or {})

        tree = {
            "layers_actual": _metric_from_agg(agg, "tree_stats.layers_actual.mean_over_seeds"),
            "num_nodes": _metric_from_agg(agg, "tree_stats.num_nodes.mean_over_seeds"),
            "num_leaves": _metric_from_agg(agg, "tree_stats.num_leaves.mean_over_seeds"),
            "avg_cluster_size": _metric_from_agg(agg, "tree_stats.avg_cluster_size.mean_over_seeds"),
        }

        retrieval = {
            "raptor_ctx_tokens": _metric_from_agg(agg, "retrieval_stats.raptor_ctx_tokens.mean_over_seeds"),
            "leaf_token_frac": _metric_from_agg(agg, "retrieval_stats.leaf_token_frac.mean_over_seeds"),
            "summary_token_frac": _metric_from_agg(agg, "retrieval_stats.summary_token_frac.mean_over_seeds"),
            "overlap_recall": _metric_from_agg(agg, "retrieval_stats.overlap_recall.mean_over_seeds"),
            # baseline_ctx_tokens is NOT aggregated by your diagnostics script, so compute from per_seed:
            "baseline_ctx_tokens": _metric_from_per_seed(
                per_seed, seeds, ["retrieval_stats", "baseline_ctx_tokens", "mean"]
            ),
            "layer_distribution_agg": agg.get("retrieval_stats.layer_distribution_agg", {}) or {},
        }

        tree_quality = {
            "intra_cluster_sim": _metric_from_agg(agg, "tree_quality.intra_cluster_sim_docmean.mean_over_seeds"),
            "inter_cluster_sim": _metric_from_agg(agg, "tree_quality.inter_cluster_sim_docmean.mean_over_seeds"),
            "parent_child_sim": _metric_from_agg(agg, "tree_quality.parent_child_sim_docmean.mean_over_seeds"),
            "compression_ratio": _metric_from_agg(agg, "tree_quality.compression_ratio_docmean.mean_over_seeds"),
        }

        skips = agg.get("skips_total_over_seeds", {}) or {}

        # Coverage info
        n_q_per_seed: Dict[str, Any] = {}
        total_q = 0
        for s in seeds:
            blk = _get_seed_block(per_seed, s)
            n_used = blk.get("n_questions_used") if isinstance(blk, dict) else None
            n_q_per_seed[str(s)] = n_used
            if isinstance(n_used, int):
                total_q += n_used

        compact["datasets"][dname] = {
            "tree": tree,
            "retrieval": {
                **retrieval,
                "n_questions_used_per_seed": n_q_per_seed,
                "n_questions_used_total": total_q,
            },
            "tree_quality": tree_quality,
            "skips_total_over_seeds": skips,
        }

    if out_path is None:
        out_path = inp_path.parent / f"summary_{inp_path.name}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None, help="Single input diagnostics JSON")
    ap.add_argument("--out", default=None, help="Optional output path (only used with --in)")
    ap.add_argument(
        "--scan-dirs",
        nargs="*",
        default=None,
        help="One or more directories to scan for diagnostics*.json files. Writes summaries next to each input.",
    )
    args = ap.parse_args()

    if not args.inp and not args.scan_dirs:
        raise SystemExit("Provide either --in <file.json> or --scan-dirs <dir1> <dir2> ...")

    if args.inp:
        inp = Path(args.inp)
        if not inp.exists():
            raise FileNotFoundError(f"Missing input: {inp}")
        outp = Path(args.out) if args.out else None
        written = summarize_one(inp, outp)
        print(f"Wrote {written}")
        return

    # scan mode
    dirs = [Path(d) for d in (args.scan_dirs or [])]
    files: List[Path] = []
    for d in dirs:
        if not d.exists():
            print(f"[warn] missing dir: {d}")
            continue
        files.extend(sorted(d.glob("diagnostics*.json")))

    if not files:
        raise SystemExit("No diagnostics*.json found in scan dirs.")

    for f in files:
        written = summarize_one(f, None)
        print(f"Wrote {written}")


if __name__ == "__main__":
    main()
