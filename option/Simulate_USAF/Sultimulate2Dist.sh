#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -p optimal
#SBATCH -A optimal

module load anaconda/2022.10
module load cuda/11.8
source activate py39

python /ailab/user/tangyuhang/LenslessFiberEndomicroscopicPhaseImaging/ASNet/TrainUnetSultimulate2Dist.py
