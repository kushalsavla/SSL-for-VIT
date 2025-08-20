#!/bin/bash
#SBATCH --job-name=ibot-classification
#SBATCH --output=ibot_classification_%j.out
#SBATCH --error=ibot_classification_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --cpus-per-task=4

echo "⏱️ Job started on: $(date)"
echo "📍 Running on node: $(hostname)"

# Load conda environment
source /work/dlclarge2/savlak-sslViT/miniconda3/etc/profile.d/conda.sh
conda activate rl_project

echo "🚀 Starting iBOT Classification on CIFAR-10"
echo "📊 Available approaches:"
echo "   1. Fine-tuning (best performance, unfreeze backbone)"
echo "   2. Linear probing (fast evaluation, frozen backbone)"
echo "   3. Nonlinear probing (balanced, frozen backbone)"
echo ""
echo "💡 Choose your approach by setting the MODE variable below!"

# Configuration - CHANGE THESE AS NEEDED
MODE="fine_tune"  # Options: fine_tune, linear_probe, nonlinear_probe
MODEL_PATH="models/ibot/final_model.pth"  # Path to iBOT pretrained model
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-4
AUGMENTATION="medium"  # Options: none, light, medium, heavy

echo "🔧 Configuration:"
echo "   Mode: $MODE"
echo "   Model: $MODEL_PATH"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Augmentation: $AUGMENTATION"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo "Available models:"
    ls -la models/ibot/
    exit 1
fi

echo "✅ Model found, starting classification..."

# Run the classification script
python -u scripts/ibot/ibot_classification.py \
    --mode $MODE \
    --model_path $MODEL_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --augmentation $AUGMENTATION

echo "✅ iBOT classification complete at: $(date)"
echo ""
echo "📊 Results saved to: results/ibot_experiments/${MODE}_results.json"
echo "💾 Best model saved to: models/ibot/best_${MODE}_model.pth"

