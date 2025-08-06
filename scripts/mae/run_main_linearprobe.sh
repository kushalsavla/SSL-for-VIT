#!/bin/bash
#SBATCH --job-name=mae-linprobe-cifar
#SBATCH --output=mae_linprobe_%j.out
#SBATCH --error=mae_linprobe_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --partition=dllabdlc_gpu-rtx2080

source /work/dlclarge2/singhk-rl_project/miniconda3/etc/profile.d/conda.sh
conda activate project

export PYTHONPATH=$PYTHONPATH:/work/dlclarge2/singhk-rl_project2/mae



python3 /work/dlclarge2/singhk-rl_project2/mae/main_linprobe.py \
  --batch_size 128 \
  --model vit_tiny_patch4_32 \
  --finetune /work/dlclarge2/singhk-rl_project2/output_dir_cifar10_pretrain/checkpoint-399.pth \
  --epochs 90 \
  --blr 0.1 \
  --input_size 32 \
  --nb_classes 10 \
  --data_path /work/dlclarge2/singhk-rl_project2/data/cifar10_splits \
  --output_dir output_dir_cifar10_linprobe_ep90
