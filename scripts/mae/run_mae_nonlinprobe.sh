#!/bin/bash
#SBATCH --job-name=mae-nonlinprobe-cifar
#SBATCH --output=mae_nonlinprobe_%j.out
#SBATCH --error=mae_nonlinprobe_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --partition=dllabdlc_gpu-rtx2080

source /work/dlclarge2/singhk-rl_project/miniconda3/etc/profile.d/conda.sh
conda activate project

export PYTHONPATH=$PYTHONPATH:/work/dlclarge2/singhk-rl_project2/mae

# Data check - fixed path typo here
ls -l /work/dlclarge2/singhk-rl_project2/data/cifar10_splits

python3 /work/dlclarge2/singhk-rl_project2/mae/main_nonlinprobe.py \
  --batch_size 128 \
  --epochs 90 \
  --model vit_tiny_patch4_32 \
  --input_size 32 \
  --patch_size 4 \
  --finetune /work/dlclarge2/singhk-rl_project2/output_dir_cifar10_pretrain/checkpoint-399.pth \
  --blr 0.01 \
  --weight_decay 0.0 \
  --warmup_epochs 10 \
  --data_path /work/dlclarge2/singhk-rl_project2/data/cifar10_splits \
  --output_dir output_dir_cifar10_nonlinprobe_ep90 \
  --num_workers 2
