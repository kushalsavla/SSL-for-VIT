#!/bin/bash
#SBATCH --job-name=vit-cifar10
#SBATCH --output=vit_cifar10_%j.out
#SBATCH --error=vit_cifar10_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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
# ✅ Run Baselines
# ----------------------------------------

export CUDA_LAUNCH_BLOCKING=1

echo "🚀 [1/3] Launching ViT Supervised Training..."
python -u train_vit.py --mode supervised
echo "✅ Supervised training complete at: $(date)"

echo "🚀 [2/3] Launching ViT Linear Probe..."
python -u train_vit.py --mode linear_probe
echo "✅ Linear probe complete at: $(date)"

echo "🚀 [3/3] Launching ViT MLP Probe..."
python -u train_vit.py --mode mlp_probe
echo "✅ MLP probe complete at: $(date)"

echo "🏁 All runs complete at: $(date)"
