#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=MakeSmallIDs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=make_small_ids_%A.out
module purge
module load 2025

source raptor_env/bin/activate

# First add IDs to full datasets

python data/processed/add_ids.py \
  --dataset quality \
  --infile data/processed/quality/eval_val.jsonl \
  --outfile data/processed/quality/eval_val_with_ids.jsonl

python data/processed/add_ids.py \
  --dataset qasper \
  --infile data/processed/qasper/eval_val.jsonl \
  --outfile data/processed/qasper/eval_val_with_ids.jsonl

python data/processed/add_ids.py \
  --dataset narrativeqa \
  --infile data/processed/narrativeqa/eval_val.jsonl \
  --outfile data/processed/narrativeqa/eval_val_with_ids.jsonl

# Then create small splits from files WITH IDs

python data/processed/make_small_splits.py \
  --dataset quality \
  --infile data/processed/quality/eval_val_with_ids.jsonl \
  --outfile data/processed/quality/eval_val_sub50_q5.jsonl \
  --max_q_per_doc 5 \
  --seed 224

python data/processed/make_small_splits.py \
  --dataset qasper \
  --infile data/processed/qasper/eval_val_with_ids.jsonl \
  --outfile data/processed/qasper/eval_val_sub50_q5.jsonl \
  --max_q_per_doc 5 \
  --seed 224

python data/processed/make_small_splits.py \
  --dataset narrativeqa \
  --infile data/processed/narrativeqa/eval_val_with_ids.jsonl \
  --outfile data/processed/narrativeqa/eval_val_sub50_q5.jsonl \
  --max_q_per_doc 5 \
  --seed 224