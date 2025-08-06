#!/bin/bash
#SBATCH --job-name=mae-finetune-cifar
#SBATCH --output=mae_finetune_%j.out
#SBATCH --error=mae_finetune_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --partition=dllabdlc_gpu-rtx2080

source /work/dlclarge2/singhk-rl_project/miniconda3/etc/profile.d/conda.sh
conda activate project

export PYTHONPATH=$PYTHONPATH:/work/dlclarge2/singhk-rl_project2/mae

ls -l /work/dlclarge2/singhk-rl_project2/data/cifar10_splits

python3 /work/dlclarge2/singhk-rl_project2/mae/main_finetune.py \
  --batch_size 128 \
  --epochs 90 \
  --model vit_tiny_patch4_32 \
  --input_size 32 \
  --patch_size 4 \
  --finetune /work/dlclarge2/singhk-rl_project2/output_dir_cifar10_pretrain/checkpoint-399.pth \
  --blr 1e-3 \
  --weight_decay 0.05 \
  --warmup_epochs 5 \
  --data_path /work/dlclarge2/singhk-rl_project2/data/cifar10_splits \
  --output_dir output_dir_cifar10_finetune_ep50 \
  --num_workers 2
