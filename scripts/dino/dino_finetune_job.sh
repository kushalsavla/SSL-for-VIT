#!/bin/bash
#SBATCH --job-name=dino-fine-tune
#SBATCH --output=dino_fine_tune_%j.out
#SBATCH --error=dino_fine_tune_%j.err
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

# Load local conda setup (using friend's environment)
source /work/dlclarge2/dragojla-workspace/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate projectenv
echo "Activated conda environment: projectenv"

# Optional: verify environment
echo " Python path: $(which python)"
python --version
pip list | grep dinov2

echo "Current working directory: $(pwd)"
ls -l

# ----------------------------------------
#  DINO Fine-tuning
# ----------------------------------------

export CUDA_LAUNCH_BLOCKING=1

echo " Launching DINO Fine-tuning..."
echo " Data path: ./data/cifar10_splits/"
echo " Model path: ./dino_vit_cifar10-200.pth"

# Run DINO fine-tuning
python -u dino_finetune.py \
    --data_path ./data/cifar10_splits/ \
    --model_path ./dino_vit_cifar10-200.pth \
    --output_dir ./dino_fine_tune_results \
    --batch_size 64 \
    --epochs 100 \
    --backbone_lr 5e-5 \
    --classifier_lr 5e-4 \
    --weight_decay 1e-4 \
    --num_workers 4

echo " DINO fine-tuning complete at: $(date)"

echo " DINO fine-tuning run complete at: $(date)"