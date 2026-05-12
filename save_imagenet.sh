#!/bin/bash
#SBATCH --job-name=save
#SBATCH --account=[Add your account]
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=47:00:00
#SBATCH --output=save_%j.out
#SBATCH --error=save_%j.err
python save_imagenet.py
