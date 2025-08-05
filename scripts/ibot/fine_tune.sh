#!/bin/bash
#SBATCH --job-name=fine-tune
#SBATCH --output=fine_tune_%j.out
#SBATCH --error=fine_tune_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --cpus-per-task=4

echo "⏱️ Job started on: $(date)"
echo "📍 Running on node: $(hostname)"
echo "🧠 Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# Load local conda setup
source /work/dlclarge2/savlak-sslViT/miniconda3/etc/profile.d/conda.sh

# Force activate rl_project environment and update PATH
conda activate rl_project
export PATH="/work/dlclarge2/savlak-sslViT/miniconda3/envs/rl_project/bin:$PATH"
echo "Activated conda environment: rl_project"

echo "Current working directory: $(pwd)"
ls -l

echo "=========================================="
echo "Running Fine-tuning for SSL-pretrained ViT-Small..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=========================================="

# Run enhanced fine-tuning with better hyperparameters
python fine_tune.py \
    --data_path ../data/cifar10_splits/ \
    --model_path ./ibot_pretrained_model/final_model.pth \
    --output_dir ./fine_tune_results \
    --batch_size 128 \
    --epochs 100 \
    --backbone_lr 5e-5 \
    --classifier_lr 5e-4 \
    --weight_decay 1e-4 \
    --num_workers 4

echo "Fine-tuning completed!"
echo "Fine-tuning completed at: $(date)" 