#!/bin/bash
#SBATCH --job-name=improved_pipeline
#SBATCH --output=improved_pipeline_%j.out
#SBATCH --error=improved_pipeline_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
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
echo "Running Improved SSL Pipeline..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=========================================="

# Step 1: Enhanced Fine-tuning
echo "🚀 Step 1: Running Enhanced Fine-tuning..."
python fine_tune.py \
    --data_path ../data/cifar10_splits/ \
    --model_path ./ibot_pretrained_model/final_model.pth \
    --output_dir ./fine_tune_results \
    --batch_size 128 \
    --epochs 100 \
    --backbone_lr 5e-5 \
    --classifier_lr 5e-4 \
    --weight_decay 1e-4 \
    --num_workers 4

if [ $? -eq 0 ]; then
    echo "✅ Enhanced fine-tuning completed successfully!"
else
    echo "❌ Enhanced fine-tuning failed!"
    exit 1
fi

# Step 2: Linear Probing on Fine-tuned Model
echo "🔬 Step 2: Running Linear Probing on Fine-tuned Model..."
python linear_probe.py \
    --data_path ../data/cifar10_splits/ \
    --model_path ./fine_tune_results/best_fine_tuned_model.pth \
    --output_dir ./linear_probe_results \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --num_workers 4

if [ $? -eq 0 ]; then
    echo "✅ Linear probing completed successfully!"
else
    echo "❌ Linear probing failed!"
    exit 1
fi

# Step 3: Non-linear Probing on Fine-tuned Model
echo "🧠 Step 3: Running Non-linear Probing on Fine-tuned Model..."
python nonlinear_probe.py \
    --data_path ../data/cifar10_splits/ \
    --model_path ./fine_tune_results/best_fine_tuned_model.pth \
    --output_dir ./nonlinear_probe_results \
    --batch_size 128 \
    --epochs 200 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --num_workers 4

if [ $? -eq 0 ]; then
    echo "✅ Non-linear probing completed successfully!"
else
    echo "❌ Non-linear probing failed!"
    exit 1
fi

# Step 4: Generate Comparison Report
echo "📊 Step 4: Generating Comparison Report..."
echo "=========================================="
echo "IMPROVED SSL PIPELINE RESULTS"
echo "=========================================="

# Read fine-tuning results
if [ -f "./fine_tune_results/fine_tune_results.txt" ]; then
    echo "📈 Fine-tuning Results:"
    grep -E "(Best Validation Accuracy|Test Accuracy)" ./fine_tune_results/fine_tune_results.txt
fi

# Read linear probing results
if [ -f "./linear_probe_results/linear_probe_results.txt" ]; then
    echo "📈 Linear Probing Results:"
    grep -E "(Best Validation Accuracy|Test Accuracy)" ./linear_probe_results/linear_probe_results.txt
fi

# Read non-linear probing results
if [ -f "./nonlinear_probe_results/nonlinear_probe_results.txt" ]; then
    echo "📈 Non-linear Probing Results:"
    grep -E "(Best Validation Accuracy|Test Accuracy)" ./nonlinear_probe_results/nonlinear_probe_results.txt
fi

echo "=========================================="
echo "Pipeline completed at: $(date)"
echo "All results saved in respective directories:"
echo "  - ./fine_tune_results/"
echo "  - ./linear_probe_results/"
echo "  - ./nonlinear_probe_results/" 