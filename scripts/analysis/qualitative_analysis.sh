#!/bin/bash
#SBATCH --job-name=qualitative-analysis
#SBATCH --output=qualitative_analysis_%j.out
#SBATCH --error=qualitative_analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --cpus-per-task=2

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
# ✅ Qualitative Analysis
# ----------------------------------------

export CUDA_LAUNCH_BLOCKING=1

# Check if image path is provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide an image path as argument"
    echo "Usage: sbatch qualitative_analysis.sh path/to/image.jpg [vit|ibot|dino]"
    exit 1
fi

IMAGE_PATH="$1"
MODEL_TYPE="${2:-vit}"  # Default to vit if not specified (options: vit, ibot, dino)

echo "🚀 Launching Qualitative Analysis..."
echo "📸 Image: $IMAGE_PATH"
echo "🤖 Model Type: $MODEL_TYPE"

# Run qualitative analysis
python -u scripts/analysis/qualitative_analysis.py \
    --image_path "$IMAGE_PATH" \
    --model_type "$MODEL_TYPE"

echo "✅ Qualitative analysis complete at: $(date)"

echo "🏁 Qualitative analysis run complete at: $(date)" 