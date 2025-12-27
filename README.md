# Raptor_ReproducibilityStudy

Reproducibility + extension study of **RAPTOR (Recursive Abstractive Processing for Tree-Based Retrieval)** for long-document QA.  
This repo contains:

- **Dataset prep**: build a small, stratified evaluation subset (50 docs, up to 5 questions/doc) for:
  - NarrativeQA
  - QuALITY
  - QASPER
- **Evaluation scripts**: run *baseline retrieval* vs *RAPTOR* under paper-aligned settings, across multiple retrieval backbones:
  - SBERT (via `eval_umaps.py`)
  - BM25 (via `bm25_raptor_retrieval.py`)
  - DPR (via `dpr_raptor_retrieval.py`)
- **Ablations/extensions**:
  - Dimensionality reduction inside clustering (UMAP / PaCMAP / TriMAP / none)
  - Alternative clustering (RAPTOR’s original GMM-style clustering vs HDBSCAN; including HDBSCAN with/without UMAP)
  - **Hierarchy depth ablation** (layer cap / tree depth) via `layer_runs.job`
- **Analysis**: posthoc diagnostics + summarization to support the “deeper analysis” section of our paper.

> Quick mental model:  
> Baselines retrieve from **leaf chunks only**.  
> RAPTOR builds a **hierarchical tree of summaries + leaves**, then retrieves from the collapsed tree under a token budget.

---

## Paper-aligned settings (used throughout)

- Leaf chunks: **100 tokens** (`raptor.utils.split_text`)
- Retrieval: **collapsed tree**, **2000-token budget**
- Reader: **UnifiedQA** (`allenai/unifiedqa-v2-t5-3b-1363200`)
  - context clipped to ~**400 tokens**, and question+context clipped to fit **≤512**
- Metrics:
  - NarrativeQA: **BLEU-1**, **BLEU-4 (equal weights)**, **ROUGE-L (F1, stemming)**, **METEOR (tokenized)**
  - QuALITY: **accuracy**
  - QASPER: **token-level F1 (SQuAD-style)**

---

## Repository structure (high level)

Typical important paths (names may differ slightly depending on branch state):

- `datasets/`
  - `prep_datasets.py` (Creates processed JSONL splits under `data/processed/<dataset>/...`)
  - `add_ids.py`  (Ensures every processed example has stable IDs (document IDs + question/example IDs) so caching and analysis can key reliably)
  - `make_small_splits.py` (Builds the evaluation subset: **50 documents** stratified by document length, with up to **5 questions per document**)
- `evaluation/`
  - `bm25_raptor_eval.py` (BM25 baseline + BM25-over-tree option)
  - `dpr_raptor_retrieval.py` (DPR retrieval backbone)
  - `eval_umaps.py` (SBERT retrieval backbone + DR + clustering ablations inside tree-building)
- `ajobs/` (HPC batch jobs on Snellius)
  - `evaluate_dr_clustering.job`
  - `analysis.job`
  - `layer_runs.job`
- `analysis/`
  - diagnostics script (tree stats, retrieval composition, overlap, etc.)
  - summarizer script (compacts diagnostics JSON into paper-ready summary JSON)
- `data/`
  - `processed/` (prepared dataset JSONL)
  - `raptor_trees/` (cached RAPTOR trees + meta)
  - `leaf_embeds/` (cached leaf embeddings per seed)

---

## Setup

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### 2) NLTK + tokenizers

Most scripts auto-download NLTK resources on first run. If you want to pre-install:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 3) OpenAI key (for RAPTOR summarization)

Tree-building uses `GPT3TurboSummarizationModel("gpt-3.5-turbo")`.

Set an API key in your environment (example):

```bash
export OPENAI_API_KEY="..."
```

If you run on Snellius, put this in your job script environment export section.

---

## Data preparation

### NarrativeQA manual download (required)

In this repo, NarrativeQA is downloaded via the official DeepMind `narrativeqa` repository and its provided story download script.

Run:

```bash
mkdir -p data/raw/narrativeqa_stories
cd data/raw/narrativeqa_stories
git clone https://github.com/deepmind/narrativeqa tmp_nqa
bash tmp_nqa/download_stories.sh
```

### Build processed JSONL splits

This creates the processed JSONL files we evaluate on (including IDs + the 50-doc subset):

```bash
python prep_datasets.py
```

Under the hood, the dataset pipeline typically does:

1. convert raw → processed JSONL
2. run `add_ids.py` to attach stable IDs
3. run `make_small_splits.py` to generate `eval_val_sub50_q5.jsonl`-style files

Outputs are typically placed under:

* `data/processed/narrativeqa/...jsonl`
* `data/processed/quality/...jsonl`
* `data/processed/qasper/...jsonl`

---

## Main evaluation: baseline vs RAPTOR

All evaluation scripts append one JSON record per run to a JSONL results file (default: `results/table_runs.jsonl`).

### SBERT backbone (baseline + RAPTOR)

```bash
# SBERT baseline (leaf-only; no tree)
python evaluation/eval_umaps.py \
  --dataset narrativeqa \
  --split data/processed/narrativeqa/eval_val_sub50_q5.jsonl \
  --seeds 224 42 99 \
  --no-with-raptor \
  --out results/table_runs.jsonl

# SBERT + RAPTOR
python evaluation/eval_umaps.py \
  --dataset narrativeqa \
  --split data/processed/narrativeqa/eval_val_sub50_q5.jsonl \
  --seeds 224 42 99 \
  --with-raptor \
  --clusterer raptor \
  --dr-method umap \
  --out results/table_runs.jsonl

```

### BM25 backbone (baseline + RAPTOR + BM25-over-tree)

`bm25_raptor_eval.py` supports:

* baseline BM25 over leaf chunks
* RAPTOR tree building (SBERT embeddings for building)
* querying the tree using **BM25** instead of cosine similarity (optional)

Examples:

```bash
# BM25 baseline (leaf-only)
python evaluation/bm25_raptor_eval.py \
  --dataset narrativeqa \
  --split data/processed/narrativeqa/eval_val_sub50_q5.jsonl \
  --retrieval-method bm25 \
  --out results/table_runs.jsonl

# RAPTOR + BM25 retrieval over the tree
python evaluation/bm25_raptor_eval.py \
  --dataset narrativeqa \
  --split data/processed/narrativeqa/eval_val_sub50_q5.jsonl \
  --with-raptor \
  --retrieval-method bm25 \
  --out results/table_runs.jsonl
```

### DPR backbone

DPR runs via:

* `evaluation/dpr_raptor_retrieval.py`

Example (pattern matches SBERT script):

```bash
python evaluation/dpr_raptor_retrieval.py \
  --dataset qasper \
  --split data/processed/qasper/eval_val_sub50_q5.jsonl \
  --with-raptor \
  --seeds 224 99 42 \
  --out results/table_runs.jsonl
```

---

## DR + clustering ablations (eval_umaps.py)

This is the “tree-building geometry matters” part of the paper: run RAPTOR with different:

* dimensionality reduction: `umap | pacmap | trimap | none`
* clusterer: `raptor | hdbscan`

Examples:

```bash
# Original-style: UMAP + RAPTOR clustering
python evaluation/eval_umaps.py \
  --dataset narrativeqa \
  --split data/processed/narrativeqa/eval_val_sub50_q5.jsonl \
  --with-raptor \
  --dr-method umap \
  --clusterer raptor \
  --seeds 224 99 42 \
  --out results/table_runs.jsonl

# HDBSCAN with no UMAP (direct clustering)
python evaluation/eval_umaps.py \
  --dataset narrativeqa \
  --split data/processed/narrativeqa/eval_val_sub50_q5.jsonl \
  --with-raptor \
  --dr-method none \
  --clusterer hdbscan \
  --seeds 224 99 42 \
  --out results/table_runs.jsonl
```

### Running on Snellius

Use:

* `ajobs/evaluate_dr_clustering.job`

This job typically sweeps the combinations and writes results to `results/table_runs.jsonl` while caching trees under `data/raptor_trees/...`.

---

## Depth ablation (hierarchy depth)

This measures sensitivity to the **max tree depth / number of layers**.
We run:

* `layer_run_narrative.py`
* `layer_run_quality.py`
* `layer_run_qasper.py`

All are orchestrated by:

* `ajobs/layer_runs.job`

Manual example (NarrativeQA):

```bash
python evaluation/layer_run_narrative.py \
  --split data/processed/narrativeqa/eval_val_sub50_q5.jsonl \
  --num-layers 3 \
  --dr-method umap \
  --clusterer raptor \
  --seeds 224 99 42 \
  --out results/layer_runs.jsonl
```

Trees for layer runs are cached in a depth-aware subdir like:
`data/raptor_trees/<dataset>/seed<seed>_<dr>_<clusterer>_L<num_layers>/`

---

## Caching (important for reproducibility + speed)

We cache **everything expensive**:

* **RAPTOR trees**: one per `(dataset, seed, document)`
  Stored under `data/raptor_trees/<dataset>/<seed_tag>.../<doc_key>.pkl`
  Plus `<doc_key>.meta.json` with builder + retriever details.
* **Leaf embeddings** (for baseline retrieval): one per `(seed, document)`
  Stored under `data/leaf_embeds/seed<seed>/<doc_key>.npy`

If you change chunking (e.g., overlap), DR method, clusterer, depth cap, etc., you should treat the caches as configuration-specific.

---

## Analysis / diagnostics (for the paper’s deeper analysis section)

There are two steps:

### 1) Generate diagnostics JSON

The diagnostics script computes (per dataset, per seed):

* tree stats (layers, nodes, leaves, avg cluster size)
* retrieval composition (leaf vs summary token fractions, layer distribution)
* optional overlap recall vs baseline
* optional “tree quality” proxies (intra/inter similarity, parent-child similarity, compression ratios)

It is typically run via:

* `ajobs/analysis.job`

Example pattern (you pass a tree subdir template like `seed{seed}_umap_raptor`):

```bash
python analysis/raptor_analysis_full.py \
  --trees-root data/raptor_trees \
  --tree-subdir-template "seed{seed}_umap_raptor" \
  --datasets narrativeqa quality qasper \
  --seeds 224 99 42 \
  --max-q 200 \
  --do-overlap \
  --baseline-embeds data/leaf_embeds \
  --out results/diagnostics_umap_raptor.json
```

### 2) Summarize diagnostics JSON into compact paper-ready JSON

This produces `summary_<diagnostics_file>.json` with the key metrics extracted + mean/std over seeds:

```bash
python analysis/summarize_diagnostics.py \
  --in results/diagnostics_umap_raptor.json
```

Scan mode (summarize all `diagnostics*.json` in a folder):

```bash
python analysis/summarize_diagnostics.py \
  --scan-dirs results/
```

---

## Outputs you should expect

* `results/table_runs.jsonl`
  Main evaluation runs (baseline and RAPTOR) for NarrativeQA / QuALITY / QASPER.
* `results/layer_runs.jsonl`
  Depth ablation runs.
* `results/diagnostics*.json`
  Rich analysis outputs used for paper plots/tables.
* `results/summary_diagnostics*.json`
  Compact summaries for easier reporting.

---

## Notes / gotchas

* **RAPTOR tree building requires an OpenAI key** (summarization).
* If you re-run with new settings (DR / clusterer / depth), keep caches separate (we do this by encoding configs into tree subdir names).
* Results are appended to JSONL files: if you want a clean run, remove or rename the output file first.

---

## Suggested “paper pipeline” order

1. `prep_datasets.py`
2. Main reproduction:

   * SBERT: `evaluation/sbert_raptor_eval.py`
   * BM25: `evaluation/bm25_raptor_eval.py`
   * DPR: `evaluation/dpr_raptor_retrieval.py`
3. DR + clustering sweeps:

   * `evaluation/eval_umaps.py` (or `ajobs/evaluate_dr_clustering.job`)
4. Depth ablation:

   * `ajobs/layer_runs.job` (runs the `layer_run_*.py` scripts)
5. Analysis:

   * `ajobs/analysis.job` → diagnostics JSON
   * `analysis/summarize_diagnostics.py` → compact summary JSON

---

## Citation

Please cite RAPTOR:

```bibtex
@inproceedings{sarthi2024raptor,
  title={RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval},
  author={Sarthi, Parth and Abdullah, Salman and Tuli, Aditi and Khanna, Shubh and Goldie, Anna and Manning, Christopher D.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

