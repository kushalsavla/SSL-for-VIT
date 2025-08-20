#!/bin/bash
#SBATCH --job-name=complete_vit_baselines
#SBATCH --output=complete_vit_baselines_%j.out
#SBATCH --error=complete_vit_baselines_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=dllabdlc_gpu-rtx2080

echo "🚀 Starting ViT Baselines Completion Pipeline..."
echo "📅 Started at: $(date)"
echo "🖥️  Running on: $(hostname)"
echo "🧠 Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# Activate conda environment
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

cd /work/dlclarge2/savlak-sslViT/SSL-for-VIT

# 🔧 CRITICAL: Fix GPU Access Issues
echo "🔧 Setting up GPU access for SLURM..."
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

# Method 1: Try GPU 0 first (most common working device)
export CUDA_VISIBLE_DEVICES=0
echo "🔧 Trying CUDA_VISIBLE_DEVICES=0 (most common working device)"

# Method 2: If SLURM specifies a different device, try that too
if [ -n "$SLURM_GPUS_ON_NODE" ] && [ "$SLURM_GPUS_ON_NODE" != "0" ]; then
    echo "⚠️  SLURM allocated GPU $SLURM_GPUS_ON_NODE, but trying GPU 0 first"
fi

# Method 3: Force CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Method 4: Set CUDA environment variables
export CUDA_LAUNCH_BLOCKING=1

echo "🔧 GPU Environment Setup Complete:"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   CUDA_DEVICE_ORDER: $CUDA_DEVICE_ORDER"
echo "   CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"

# 🔍 Verify GPU Access with Fallback
echo "🔍 Verifying GPU access..."
python -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name()}')
    print('✅ GPU access verified successfully!')
else:
    print('❌ GPU 0 access failed, trying GPU 1...')
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.cuda.empty_cache()
    print(f'PyTorch CUDA available after GPU 1: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU count: {torch.cuda.device_count()}')
        print(f'Current GPU: {torch.cuda.current_device()}')
        print(f'GPU name: {torch.cuda.get_device_name()}')
        print('✅ GPU 1 access verified successfully!')
    else:
        print('❌ Both GPU 0 and GPU 1 failed!')
        print('💡 This suggests a deeper SLURM GPU isolation issue')
        exit(1)
"

echo "📊 Phase 1: ViT Random Linear Probing..."
cd scripts/experiments/vit
python vit_random_linear_probe.py \
    --output_dir /work/dlclarge2/savlak-sslViT/SSL-for-VIT/models/experiments/vit_random_linear_probe \
    --data_path /work/dlclarge2/savlak-sslViT/SSL-for-VIT/data/cifar10_splits \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.01

echo "📊 Phase 2: ViT Random Nonlinear Probing..."
python vit_random_nonlinear_probe.py \
    --output_dir /work/dlclarge2/savlak-sslViT/SSL-for-VIT/models/experiments/vit_random_nonlinear_probe \
    --data_path /work/dlclarge2/savlak-sslViT/SSL-for-VIT/data/cifar10_splits \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.01

echo "📊 Phase 3: ViT Finetuning (using existing trained model)..."
python vit_finetune.py \
    --output_dir /work/dlclarge2/savlak-sslViT/SSL-for-VIT/models/experiments/vit_finetune \
    --data_path /work/dlclarge2/savlak-sslViT/SSL-for-VIT/data/cifar10_splits \
    --pretrained_path /work/dlclarge2/savlak-sslViT/SSL-for-VIT/models/experiments/vit_supervised/best_vit_supervised.pth \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4

cd /work/dlclarge2/savlak-sslViT/SSL-for-VIT

echo "✅ ViT Baselines Pipeline Completed!"
echo "📅 Finished at: $(date)"

# Save completion status
echo "COMPLETED" > models/experiments/vit_baselines_status.txt
echo "$(date)" >> models/experiments/vit_baselines_status.txt
