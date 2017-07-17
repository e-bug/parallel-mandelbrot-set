#!/bin/bash
# SBATCH --reservation phpc2017
# SBATCH --account phpc2017
# SBATCH -N 1
# SBATCH -n 1
# SBATCH -t 08:00:00


./opti2_escape_time o2  1920  1080 10000
./opti2_escape_time o2  2560  1440 10000
./opti2_escape_time o2  3840  2160 10000
./opti2_escape_time o2 10240  5760 10000
./opti2_escape_time o2 12800  7200 10000
./opti2_escape_time o2 16000  9000 10000
./opti2_escape_time o2 19200 10800 10000
./opti2_escape_time o2 25600 14400 10000

./opti2_escape_time o2 25600 14400     1000
./opti2_escape_time o2 25600 14400    10000
./opti2_escape_time o2 25600 14400   100000
./opti2_escape_time o2 25600 14400  1000000
./opti2_escape_time o2 25600 14400 10000000