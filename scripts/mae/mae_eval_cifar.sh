#!/bin/bash
#SBATCH --job-name=mae-eval-cifar10
#SBATCH --output=mae_eval_%j.out
#SBATCH --error=mae_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --partition=dllabdlc_gpu-rtx2080

# Activate your conda environment
source /work/dlclarge2/singhk-rl_project/miniconda3/etc/profile.d/conda.sh
conda activate project

export PYTHONPATH=$PYTHONPATH:/work/dlclarge2/singhk-rl_project2/mae

# Optional: Sanity check for data and checkpoint locations
ls -l /work/dlclarge2/singhk-rl_project2/data/cifar10_splits
ls -lh /work/dlclarge2/singhk-rl_project2/mae/output_dir_cifar10_finetune_ep50

# Run the evaluation script
python3 /work/dlclarge2/singhk-rl_project2/mae/test_unseen_data.py \
  --data_path /work/dlclarge2/singhk-rl_project2/data/cifar10_splits \
  --model_path /work/dlclarge2/singhk-rl_project2/output_dir_cifar10_finetune_ep50/checkpoint-89.pth \
  --batch_size 128 \
  --num_workers 2
