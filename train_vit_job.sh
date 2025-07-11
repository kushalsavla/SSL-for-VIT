#!/bin/bash
#SBATCH --job-name=vit-cifa
#SBATCH --output=vit_cifar10_%j.out
#SBATCH --error=vit_cifar10_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --partition=dllabdlc_gpu-rtx2080

# Load your local conda setup
source /work/dlclarge2/savlak-sslViT/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate rl_project

# Optional: confirm environment
which python3
python3 --version
pip list | grep dinov2

# Run training script
python -u train_vit.py
