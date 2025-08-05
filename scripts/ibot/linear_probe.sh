#!/bin/bash
#SBATCH --job-name=linear-probe
#SBATCH --output=linear_probe_%j.out
#SBATCH --error=linear_probe_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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
echo "Running Linear Probing for SSL-pretrained ViT-Small..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=========================================="

# Run linear probing
python linear_probe.py \
    --data_path ../data/cifar10_splits/ \
    --model_path ./ibot_pretrained_model/final_model.pth \
    --output_dir ./linear_probe_results \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --num_workers 4

echo "Linear probing completed!"
echo "Linear probing completed at: $(date)" 