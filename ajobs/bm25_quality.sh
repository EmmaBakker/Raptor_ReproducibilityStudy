#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=bm25qualityRaptorNoRaptor
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=quality_bm25_without_with_raptor_%A.out

module purge
module load 2025

#python -m venv raptor_env


source raptor_env/bin/activate

source config/secrets.env

PYTHONPATH=. python evaluation/sbert_raptor_eval.py --dataset quality --split data/processed/quality/eval_val_sub50_q5.jsonl --retrieval-method bm25

PYTHONPATH=. python evaluation/sbert_raptor_eval.py --dataset quality --split data/processed/quality/eval_val_sub50_q5.jsonl --retrieval-method bm25 --with-raptor

# PYTHONPATH=. python evaluation/sbert_raptor_eval.py --dataset quality --split data/processed/quality/eval_val.jsonl --retrieval-method bm25

# PYTHONPATH=. python evaluation/sbert_raptor_eval.py --dataset quality --split data/processed/quality/eval_val.jsonl --retrieval-method bm25 --with-raptor