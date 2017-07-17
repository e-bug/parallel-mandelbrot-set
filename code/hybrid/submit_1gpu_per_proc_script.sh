#!/bin/bash
#SBATCH --partition gpu
#SBATCH --qos gpu_free
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --gres gpu:2


module load gcc cuda mvapich2
rm opti5*_escape_time_1gpu_per_proc
make

echo -e "\n\n"
srun ./opti5_escape_time_1gpu_per_proc o5  1920  1080 10000
srun ./opti5_escape_time_1gpu_per_proc o5  2560  1440 10000   
srun ./opti5_escape_time_1gpu_per_proc o5  3840  2160 10000   
srun ./opti5_escape_time_1gpu_per_proc o5 10240  5760 10000   
srun ./opti5_escape_time_1gpu_per_proc o5 12800  7200 10000   
srun ./opti5_escape_time_1gpu_per_proc o5 16000  9000 10000  
srun ./opti5_escape_time_1gpu_per_proc o5 19200 10800 10000  
srun ./opti5_escape_time_1gpu_per_proc o5 25600 14400 10000
echo -e "\n"
srun ./opti5_escape_time_1gpu_per_proc o5 25600 14400     1000
srun ./opti5_escape_time_1gpu_per_proc o5 25600 14400    10000
srun ./opti5_escape_time_1gpu_per_proc o5 25600 14400   100000
srun ./opti5_escape_time_1gpu_per_proc o5 25600 14400  1000000
srun ./opti5_escape_time_1gpu_per_proc o5 25600 14400 10000000

echo -e "\n\n"
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO  1920  1080 10000
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO  2560  1440 10000
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO  3840  2160 10000
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO 10240  5760 10000
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO 12800  7200 10000
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO 16000  9000 10000
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO 19200 10800 10000
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO 25600 14400 10000
echo -e "\n"
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO 25600 14400     1000
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO 25600 14400    10000
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO 25600 14400   100000
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO 25600 14400  1000000
srun ./opti5-IO_escape_time_1gpu_per_proc o5IO 25600 14400 10000000

