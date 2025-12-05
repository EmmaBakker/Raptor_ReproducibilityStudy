#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=install_env_%A.out

module purge
module load 2025

source raptor_env/bin/activate

# mkdir -p data/raw/narrativeqa_stories
# cd data/raw/narrativeqa_stories
# git clone https://github.com/deepmind/narrativeqa tmp_nqa
# chmod +x tmp_nqa/download_stories.sh
# ./tmp_nqa/download_stories.sh
# cd ../../../..

python -m data.processed.prep_datasets --datasets