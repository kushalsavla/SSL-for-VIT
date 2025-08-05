#!/bin/bash
#SBATCH --job-name=dino-cifar10-pretrain
#SBATCH --output=dino_cifar10_pretrain_%j.out
#SBATCH --error=dino_cifar10_pretrain_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --partition=dllabdlc_gpu-rtx2080

# Load your local conda setup
source /work/dlclarge2/dragojla-workspace/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate projectenv

# Optional: confirm environment
which python3
python3 --version
pip list | grep dinov2

# Run training script
python -u pretrain_dino.py
