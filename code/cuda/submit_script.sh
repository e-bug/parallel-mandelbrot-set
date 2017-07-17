#!/bin/bash
#SBATCH --partition gpu
#SBATCH --qos gpu_free
#SBATCH --gres gpu:1
#SBATCH -N 1
#SBATCH -n 1

module load gcc cuda
rm opti3_escape_time opti4_escape_time opti5_escape_time
make

srun ./opti3_escape_time o3 25600 14400 1000000
srun ./opti4_escape_time o4 25600 14400 1000000
srun ./opti5_escape_time o5 25600 14400 1000000

echo -e "\n"
