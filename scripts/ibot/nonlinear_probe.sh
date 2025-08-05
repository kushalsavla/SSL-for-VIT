#!/bin/bash
#SBATCH --job-name=nonlinear-probe
#SBATCH --output=nonlinear_probe_%j.out
#SBATCH --error=nonlinear_probe_%j.err
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
echo "Running Non-Linear Probing for SSL-pretrained ViT-Small..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=========================================="

# Run non-linear probing
python nonlinear_probe.py \
    --data_path ../data/cifar10_splits/ \
    --model_path ./ibot_pretrained_model/final_model.pth \
    --output_dir ./nonlinear_probe_results \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --hidden_dims 512 256 \
    --num_workers 4

echo "Non-linear probing completed!"
echo "Non-linear probing completed at: $(date)" 