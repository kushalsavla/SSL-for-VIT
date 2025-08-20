#!/bin/bash
#SBATCH --job-name=vit-finetune-anti-memorization
#SBATCH --output=vit_finetune_anti_memorization_%j.out
#SBATCH --error=vit_finetune_anti_memorization_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --cpus-per-task=4

echo "⏱️ Job started on: $(date)"
echo "📍 Running on node: $(hostname)"

# Load conda environment
source /work/dlclarge2/savlak-sslViT/miniconda3/etc/profile.d/conda.sh
conda activate rl_project

echo "🚀 Starting ANTI-MEMORIZATION ViT Fine-tuning on CIFAR-10"
echo "📊 Anti-memorization improvements:"
echo "   • Stronger regularization (dropout 0.3, weight decay 0.1)"
echo "   • Enhanced augmentation (Mixup + CutMix)"
echo "   • Early stopping (patience 15)"
echo "   • Lower learning rate (5e-5) for fine-tuning"
echo "   • Reduced training epochs (100 max)"
echo "💡 Goal: Learn generalizable features, not memorize training data!"

python -u scripts/vit/finetune_vit.py \
    --epochs 100 \
    --batch_size 32 \
    --lr 5e-5

echo "✅ Anti-memorization fine-tuning complete at: $(date)"
