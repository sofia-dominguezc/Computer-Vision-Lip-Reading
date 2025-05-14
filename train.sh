#!/bin/bash
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mem=5G
#SBATCH --gres=gpu:a100:1
python train.py --exp-name=combined_1 --checkpoint-path exp_dir/vsr_trlrs2lrs3vox2avsp_base.pth --batch-size 8 --epochs 10 --warmup-epochs 5 --lr 5e-6 --freeze-layers --debug