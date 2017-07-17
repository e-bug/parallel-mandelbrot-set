#!/bin/bash
#SBATCH --partition gpu
#SBATCH --qos gpu_free
#SBATCH --gres gpu:1
#SBATCH -N 1
#SBATCH -n 1

module load gcc cuda
rm opti5_escape_time
make

nvprof --print-gpu-trace ./opti5_escape_time o5 2560 1440 1000000
echo -e "\n"
nvprof --metrics achieved_occupancy ./opti5_escape_time o5 2560 1440 1000000
nvprof --metrics achieved_occupancy ./opti5_escape_time o5 25600 14400 1000000

echo -e "\n"
