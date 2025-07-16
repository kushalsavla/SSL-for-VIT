#!/bin/bash
#SBATCH --job-name=dino-cifar10
#SBATCH --output=dino_cifar10_%j.out
#SBATCH --error=dino_cifar10_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=dllabdlc_gpu-rtx2080

# ----------------------------------------
# ✅ Environment Setup
# ----------------------------------------             
echo "⏱️ Job started on: $(date)"
echo "📍 Running on node: $(hostname)"
echo "🧠 Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# Load conda environment
source /work/dlclarge2/savlak-sslViT/miniconda3/etc/profile.d/conda.sh
if [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
    conda activate rl_project
    echo "✅ Switched to conda environment: rl_project"
else
    echo "✅ Using current conda environment: $CONDA_DEFAULT_ENV"
fi

# Info check
echo "🐍 Python path: $(which python)"
python --version
pip list | grep dinov2
echo "📂 Current working directory: $(pwd)"
ls -l

# ----------------------------------------
# ✅ Run DINO Pretraining
# ----------------------------------------

export CUDA_LAUNCH_BLOCKING=1

echo "🚀 Starting DINO pretraining with epoch timing..."

# Run the Python script and time each epoch
python3 -u train_dino.py --epochs 300 --batch-size 128

echo "✅ DINO pretraining complete at: $(date)"
echo "🏁 Job finished at: $(date)"
