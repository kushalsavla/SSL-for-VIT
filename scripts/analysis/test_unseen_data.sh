#!/bin/bash
#SBATCH --job-name=test_unseen
#SBATCH --output=test_unseen_%j.out
#SBATCH --error=test_unseen_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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
echo "Testing ViT, DINO, and iBOT on Unseen Data..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=========================================="

# Run test on unseen data
python scripts/analysis/test_unseen_data.py \
    --data_path ./data/cifar10_splits/ \
    --vit_model_path ./models/vit/best_vit_small_model.pth \
    --dino_model_path ./models/dino/best_dino_fine_tuned_model.pth \
    --ibot_model_path ./models/ibot/best_fine_tuned_model.pth \
    --batch_size 128 \
    --num_workers 4

echo "Testing completed!"
echo "Testing completed at: $(date)" 