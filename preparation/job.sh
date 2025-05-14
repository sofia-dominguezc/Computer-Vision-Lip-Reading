#!/bin/bash
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mem=50G
#SBATCH -p cpl
#SBATCH --gres=gpu:a100:1
python preprocess_tcd-timit.py --data-dir data_dir/TCD-TIMIT --root-dir ../tcd-timit --dataset tcd-timit