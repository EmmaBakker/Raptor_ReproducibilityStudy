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

#python -m venv raptor_env
source raptor_env/bin/activate
pip install -r requirements.txt