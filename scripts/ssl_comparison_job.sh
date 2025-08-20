#!/bin/bash
#SBATCH --job-name=ssl-comparison
#SBATCH --output=ssl_comparison_%j.out
#SBATCH --error=ssl_comparison_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --cpus-per-task=4

echo "⏱️ Job started on: $(date)"
echo "📍 Running on node: $(hostname)"

# Load conda environment
source /work/dlclarge2/savlak-sslViT/miniconda3/etc/profile.d/conda.sh
conda activate rl_project

echo "🚀 Starting Comprehensive SSL Comparative Study"
echo "📊 This will compare:"
echo "   • iBOT: Linear & Nonlinear probing"
echo "   • DINO: Linear & Nonlinear probing"
echo "   • MAE: Linear & Nonlinear probing"
echo ""
echo "🎯 Goal: Compare representation quality across SSL methods"
echo "💡 This addresses Milestone 3 of your RL#2 project!"

# Check available models
echo "🔍 Checking available models..."
echo "iBOT models:"
ls -la models/ibot/ 2>/dev/null || echo "   No iBOT models found"
echo ""
echo "DINO models:"
ls -la models/dino/ 2>/dev/null || echo "   No DINO models found"
echo ""
echo "MAE models:"
ls -la models/mae/ 2>/dev/null || echo "   No MAE models found"
echo ""

# Run the comprehensive comparison
echo "🚀 Running SSL comparison experiments..."
python -u scripts/comparative_ssl_study.py --run_experiments

echo "📊 Collecting and displaying results..."
python -u scripts/comparative_ssl_study.py --collect_results

echo "✅ SSL comparison study complete at: $(date)"
echo ""
echo "📋 Summary:"
echo "   • Results saved to: results/ssl_comparison_report.json"
echo "   • Individual results in: results/*_experiments/"
echo "   • This provides the comparative analysis needed for your RL#2 project!"
