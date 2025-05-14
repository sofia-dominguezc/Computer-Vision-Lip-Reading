#!/bin/bash
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:a100:1
python domain_adapt.py