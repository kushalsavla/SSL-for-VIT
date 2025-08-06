#!/bin/bash
#SBATCH --job-name=comprehensive_analysis
#SBATCH --output=comprehensive_analysis_%j.out
#SBATCH --error=comprehensive_analysis_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=dllabdlc_gpu-rtx2080

echo "⏱️ Job started on: $(date)"
echo "📍 Running on node: $SLURM_NODELIST"
echo "🧠 Allocated GPU(s): $CUDA_VISIBLE_DEVICES"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_project
echo "Switched to conda environment: rl_project"

# Set Python path
export PYTHONPATH="/work/dlclarge2/savlak-sslViT/SSL-for-VIT:$PYTHONPATH"
echo "🐍 Python path: $(which python)"
python --version

# Set working directory
cd /work/dlclarge2/savlak-sslViT/SSL-for-VIT
echo "Current working directory: $(pwd)"
ls -la

echo "🚀 Launching Comprehensive Analysis..."
python comprehensive_analysis.py

echo "✅ Comprehensive analysis complete at: $(date)"
echo "🏁 Comprehensive analysis run complete at: $(date)" 