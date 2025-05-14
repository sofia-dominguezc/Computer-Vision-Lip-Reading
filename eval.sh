#!/bin/bash
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:a100:1
python eval.py --test-root-dir testset --checkpoint-path=exp_dir/combined_1/last.ckpt --beam-size=20 --lm-weight 0.1 --lm-name gemma-domain-adapt --debug