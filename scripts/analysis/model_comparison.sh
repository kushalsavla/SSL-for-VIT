#!/bin/bash
#SBATCH --job-name=model_comparison
#SBATCH --output=outputs/model_comparison_%j.out
#SBATCH --error=outputs/model_comparison_%j.err
#SBATCH --time=01:00:00
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

# Create outputs directory if it doesn't exist
mkdir -p outputs

echo "🚀 Launching Model Comparison Analysis..."
echo "📊 Mode: Both single-image and multi-image analysis"

# Run model comparison (both single and multi-image analysis)
python scripts/analysis/model_comparison.py --mode both --num_samples 100

echo "✅ Model comparison complete at: $(date)"
echo "📊 Results saved to: results/final_evaluation/"
echo "🏁 Model comparison run complete at: $(date)" 