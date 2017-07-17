#!/bin/bash
#SBATCH --partition gpu
#SBATCH --qos gpu_free
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --gres gpu:4


module load gcc cuda mvapich2
rm opti5*_escape_time_2gpu_per_proc
make

echo -e "\n\n"
srun ./opti5_escape_time_2gpu_per_proc o52  1920  1080 10000
srun ./opti5_escape_time_2gpu_per_proc o52  2560  1440 10000
srun ./opti5_escape_time_2gpu_per_proc o52  3840  2160 10000
srun ./opti5_escape_time_2gpu_per_proc o52 10240  5760 10000
srun ./opti5_escape_time_2gpu_per_proc o52 12800  7200 10000
srun ./opti5_escape_time_2gpu_per_proc o52 16000  9000 10000
srun ./opti5_escape_time_2gpu_per_proc o52 19200 10800 10000
srun ./opti5_escape_time_2gpu_per_proc o52 25600 14400 10000
echo -e "\n"
srun ./opti5_escape_time_2gpu_per_proc o52 25600 14400     1000
srun ./opti5_escape_time_2gpu_per_proc o52 25600 14400    10000
srun ./opti5_escape_time_2gpu_per_proc o52 25600 14400   100000
srun ./opti5_escape_time_2gpu_per_proc o52 25600 14400  1000000
srun ./opti5_escape_time_2gpu_per_proc o52 25600 14400 10000000

echo -e "\n\n"
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO  1920  1080 10000
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO  2560  1440 10000
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO  3840  2160 10000
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO 10240  5760 10000
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO 12800  7200 10000
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO 16000  9000 10000
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO 19200 10800 10000
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO 25600 14400 10000
echo -e "\n"
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO 25600 14400     1000
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO 25600 14400    10000
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO 25600 14400   100000
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO 25600 14400  1000000
srun ./opti5-IO_escape_time_2gpu_per_proc o52IO 25600 14400 10000000

