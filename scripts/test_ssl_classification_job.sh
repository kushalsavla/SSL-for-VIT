#!/bin/bash
#SBATCH --job-name=ssl-classification-test
#SBATCH --output=ssl_classification_test_%j.out
#SBATCH --error=ssl_classification_test_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
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

echo "Current working directory: $(pwd)"
ls -l

# ----------------------------------------
# ✅ SSL Classification Testing
# ----------------------------------------

export CUDA_LAUNCH_BLOCKING=1

echo "🚀 Starting SSL Classification Testing"
echo "📊 This will test all SSL methods on CIFAR-10 classification"
echo "💡 Goal: Evaluate representation quality of SSL methods!"

cd scripts/experiments

# Test MAE Linear Probing
echo "🔍 Testing MAE Linear Probing..."
python -u test_ssl_classification.py \
    --method mae \
    --model_path ../../models/experiments/mae_pretrain/checkpoint-199.pth \
    --probe_type linear \
    --epochs 100

# Test MAE Nonlinear Probing
echo "🔍 Testing MAE Nonlinear Probing..."
python -u test_ssl_classification.py \
    --method mae \
    --model_path ../../models/experiments/mae_pretrain/checkpoint-199.pth \
    --probe_type nonlinear \
    --epochs 100

# Test DINO Linear Probing
echo "🔍 Testing DINO Linear Probing..."
python -u test_ssl_classification.py \
    --method dino \
    --model_path ../../models/experiments/dino_pretrain/checkpoint-199.pth \
    --probe_type linear \
    --epochs 100

# Test DINO Nonlinear Probing
echo "🔍 Testing DINO Nonlinear Probing..."
python -u test_ssl_classification.py \
    --method dino \
    --model_path ../../models/experiments/dino_pretrain/checkpoint-199.pth \
    --probe_type nonlinear \
    --epochs 100

# Test iBOT Linear Probing
echo "🔍 Testing iBOT Linear Probing..."
python -u test_ssl_classification.py \
    --method ibot \
    --model_path ../../models/experiments/ibot_pretrain/checkpoint-199.pth \
    --probe_type linear \
    --epochs 100

# Test iBOT Nonlinear Probing
echo "🔍 Testing iBOT Nonlinear Probing..."
python -u test_ssl_classification.py \
    --method ibot \
    --model_path ../../models/experiments/ibot_pretrain/checkpoint-199.pth \
    --probe_type nonlinear \
    --epochs 100

echo "✅ All SSL classification testing complete at: $(date)"
echo "💾 Results saved to: ../../results/ssl_experiments/"
echo "📊 Check individual result files for detailed performance metrics"
