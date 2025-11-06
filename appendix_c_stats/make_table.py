# appendix_c_make_table.py
import csv, math
from pathlib import Path
from collections import defaultdict

def summarize_weighted(rows):
    agg = {"parents":0, "sum_children":0.0, "sum_child_len_sum":0.0, "sum_summary_len":0.0}
    for r in rows:
        n = int(r["nodes"])
        agg["parents"] += n
        agg["sum_children"] += float(r["avg_children"]) * n
        agg["sum_child_len_sum"] += float(r["avg_child_len_sum"])           # already a TOTAL
        agg["sum_summary_len"] += float(r["avg_summary_len"]) * n            # per-parent → total
    if agg["parents"] == 0: return None
    avg_children = agg["sum_children"] / agg["parents"]
    avg_child_len_per_child = (agg["sum_child_len_sum"] / agg["sum_children"]) if agg["sum_children"]>0 else 0.0
    avg_summary_len = agg["sum_summary_len"] / agg["parents"]
    compression = (agg["sum_summary_len"] / agg["sum_child_len_sum"]) if agg["sum_child_len_sum"]>0 else 0.0
    return avg_summary_len, avg_child_len_per_child, avg_children, compression

def load_rows(csv_path):
    rows=[]
    with open(csv_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

base = Path("appendix_c_stats")
datasets = {
    "QuALITY":     base / "stats_quality.csv",
    "NarrativeQA": base / "stats_narrativeqa.csv",
    "QASPER":      base / "stats_qasper.csv",
}

table = {}
all_rows = []
for name, path in datasets.items():
    rows = load_rows(path)
    all_rows.extend(rows)
    s = summarize_weighted(rows)
    if s:
        table[name] = s

table["All Datasets"] = summarize_weighted(all_rows)

# Pretty print; round like the paper
print(f"{'Dataset':<13}  {'Avg. Summary Len':>18}  {'Avg. Child Len':>15}  {'Avg. # Children':>16}  {'Avg. Compression':>17}")
for k in ["All Datasets","QuALITY","NarrativeQA","QASPER"]:
    a = table[k]
    if not a: continue
    avg_sum, avg_child, avg_kids, comp = a
    print(f"{k:<13}  {avg_sum:18.1f}  {avg_child:15.1f}  {avg_kids:16.1f}  {comp:17.2f}")
