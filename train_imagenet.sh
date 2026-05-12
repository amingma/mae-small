#!/bin/bash
#SBATCH --job-name=save
#SBATCH --account=[Add your account]
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
python vit_base.py
