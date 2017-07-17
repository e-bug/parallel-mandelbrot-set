#!/bin/bash
# SBATCH --reservation phpc2017
# SBATCH --account phpc2017
# SBATCH -N 1
# SBATCH -n 1
# SBATCH -t 12:00:00


./naive_smooth_escape_time o0  1920  1080 10000
./naive_smooth_escape_time o0  2560  1440 10000
./naive_smooth_escape_time o0  3840  2160 10000
./naive_smooth_escape_time o0 10240  5760 10000
./naive_smooth_escape_time o0 12800  7200 10000
./naive_smooth_escape_time o0 16000  9000 10000
./naive_smooth_escape_time o0 19200 10800 10000
./naive_smooth_escape_time o0 25600 14400 10000

./naive_smooth_escape_time o0 25600 14400     1000
./naive_smooth_escape_time o0 25600 14400    10000
./naive_smooth_escape_time o0 25600 14400   100000
./naive_smooth_escape_time o0 25600 14400  1000000
./naive_smooth_escape_time o0 25600 14400 10000000
