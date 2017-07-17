#!/bin/bash
#SBATCH --partition gpu
#SBATCH --qos gpu_free
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --gres gpu:4


module load gcc cuda mvapich2
rm opti5*_escape_time_*gpu_per_proc
make

echo -e "\n\n"
srun ./opti5_escape_time_1gpu_per_proc o5
srun ./opti5_escape_time_2gpu_per_proc o52

echo -e "\n\n"
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO
