#!/bin/bash
#SBATCH --job-name=vit-nonlinear-probe
#SBATCH --output=vit_nonlinear_probe_%j.out
#SBATCH --error=vit_nonlinear_probe_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --cpus-per-task=4

# ----------------------------------------
# ✅ Environment Setup
# ----------------------------------------

echo "⏱️ Job started on: $(date)"
echo "📍 Running on node: $(hostname)"
echo "🧠 Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# Load local conda setup
source /work/dlclarge2/savlak-sslViT/miniconda3/etc/profile.d/conda.sh
# If current environment is 'base', activate 'rl_project' (which has all dependencies)
if [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
conda activate rl_project
    echo "Switched to conda environment: rl_project"
else
    echo "Using current conda environment: $CONDA_DEFAULT_ENV"
fi

# Optional: verify environment
echo "🐍 Python path: $(which python)"
python --version
pip list | grep dinov2

echo "Current working directory: $(pwd)"
ls -l

# ----------------------------------------
# ✅ Run Non-linear Probing
# ----------------------------------------

export CUDA_LAUNCH_BLOCKING=1

echo "🚀 Launching ViT Non-linear Probing..."
python -u nonlinear_probe_vit.py \
    --data_path ../data/cifar10_splits/ \
    --model_path ./best_vit_small_model.pth \
    --output_dir ./nonlinear_probe_results \
    --batch_size 128 \
    --epochs 200 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --dropout 0.2 \
    --hidden_dims 512 256 \
    --num_workers 4
echo "✅ Non-linear probing complete at: $(date)"

echo "🏁 Non-linear probing run complete at: $(date)" 