#!/bin/bash
#SBATCH --job-name=mae-pretrain-cifar
#SBATCH --output=mae_pretrain_%j.out
#SBATCH --error=mae_pretrain_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --partition=dllabdlc_gpu-rtx2080

source /work/dlclarge2/singhk-rl_project/miniconda3/etc/profile.d/conda.sh
conda activate project

export PYTHONPATH=$PYTHONPATH:/work/dlclarge2/singhk-rl_project2/mae

# Check your .npy files are in this directory
ls -l /work/dlclarge2/singhk-rl_project2/data/cifar10_splits

python3 /work/dlclarge2/singhk-rl_project2/mae/main_pretrain.py \
  --batch_size 128 \
  --epochs 400 \
  --model mae_vit_tiny_patch4_32 \
  --input_size 32 \
  --mask_ratio 0.75 \
  --blr 1e-3 \
  --weight_decay 0.05 \
  --warmup_epochs 40 \
  --data_path /work/dlclarge2/singhk-rl_project2/data/cifar10_splits \
  --output_dir output_dir_cifar10_pretrain \
  --num_workers 2
