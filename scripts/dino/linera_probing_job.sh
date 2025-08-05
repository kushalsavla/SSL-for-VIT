#!/bin/bash
#SBATCH --job-name=dino-cifar10
#SBATCH --output=dino_cifar10_%j.out
#SBATCH --error=dino_cifar10_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --cpus-per-task=4

# ----------------------------------------
#  Environment Setup
# ----------------------------------------

echo " Job started on: $(date)"
echo " Running on node: $(hostname)"
echo " Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# Load local conda setup
source /work/dlclarge2/dragojla-workspace/miniconda3/etc/profile.d/conda.sh
# If current environment is 'base', activate 'activate projectenv' (which has all dependencies)
if [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
conda activate activate projectenv
    echo "Switched to conda environment:activate projectenv"
else
    echo "Using current conda environment: $CONDA_DEFAULT_ENV"
fi

# Optional: verify environment
echo " Python path: $(which python)"
python --version
pip list | grep dinov2

echo "Current working directory: $(pwd)"
ls -l

# ----------------------------------------
#  Run Baselines
# ----------------------------------------

export CUDA_LAUNCH_BLOCKING=1

echo " Linear probing"
python -u linear_probing.py 
echo " Supervised training complete at: $(date)"


echo " All runs complete at: $(date)"