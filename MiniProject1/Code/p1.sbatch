#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=p1
#SBATCH --output=p1.out

module purge
module load anaconda3/2020.07
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate ~/penv
export PATH=~/penv/bin:$PATH

python ~/code/p1.py --bs 128 --lr 0.1 --decay_step 40