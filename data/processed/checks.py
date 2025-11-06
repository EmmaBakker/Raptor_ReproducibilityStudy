import json, pathlib

def count_jsonl(path):
    return sum(1 for _ in open(path, 'r', encoding='utf-8'))

for name in ["qasper","quality","narrativeqa"]:
    base = pathlib.Path("data/processed")/name
    print(f"\n== {name.upper()} ==")
    for split in ["train","val","test"]:
        d = base/f"docs_{split}.jsonl"
        q = base/f"qa_{split}.jsonl"
        if d.exists():
            print(split, "docs:", count_jsonl(d))
        if q.exists():
            print(split, "qas :", count_jsonl(q))
