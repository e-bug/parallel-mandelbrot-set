#!/bin/bash
# SBATCH --reservation phpc2017
# SBATCH --account phpc2017
# SBATCH -N 1
# SBATCH -n 1
# SBATCH -t 10:00:00


./opti1_escape_time o1  1920  1080 10000
./opti1_escape_time o1  2560  1440 10000
./opti1_escape_time o1  3840  2160 10000
./opti1_escape_time o1 10240  5760 10000
./opti1_escape_time o1 12800  7200 10000
./opti1_escape_time o1 16000  9000 10000
./opti1_escape_time o1 19200 10800 10000
./opti1_escape_time o1 25600 14400 10000

./opti1_escape_time o1 25600 14400     1000
./opti1_escape_time o1 25600 14400    10000
./opti1_escape_time o1 25600 14400   100000
./opti1_escape_time o1 25600 14400  1000000
./opti1_escape_time o1 25600 14400 10000000
