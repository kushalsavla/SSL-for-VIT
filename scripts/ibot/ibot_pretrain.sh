#!/bin/bash
#SBATCH --job-name=ultra-simple-ibot
#SBATCH --output=ultra_simple_ibot_%j.out
#SBATCH --error=ultra_simple_ibot_%j.err
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
echo "Running iBOT Pretraining for CIFAR-10..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=========================================="

# Run iBOT pretraining
python ibot_pretrain.py \
    --data_path ../data/cifar10_splits/ \
    --output_dir ./ultra_simple_output \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.0005 \
    --weight_decay 0.04 \
    --saveckp_freq 10 \
    --num_workers 4 \
    --warmup_epochs 10 \
    --min_lr 1e-6 \
    --momentum_teacher 0.996 \
    --seed 42

echo "iBOT pretraining completed!"
echo "Training completed at: $(date)" 