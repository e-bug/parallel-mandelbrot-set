#!/bin/bash
# SBATCH --reservation phpc2017
# SBATCH --account phpc2017
# SBATCH -N 1
# SBATCH -n 1


# For each optimization level and for each compiler (gcc and intel): 
# 1. create all executables with the -g -pg flags (for profiling)
# 2. run each executable producing an image with the optimization level and
#    the compiler used (e.g. _O0gcc means that gcc and -O0 were used)
# 3. rename the special file gmon.out produced by each executable and the 
#    executable to reflect which compier and optimization levels were used

args="25600 14400 10000"

module load gcc
rm opti3_escape_time
# ============================================================================ #
#                                                                              #
#                                      O0                                      #
#                                                                              #
# ============================================================================ #
make CC="gcc" CFLAGS="-g -pg -O0"

./opti3_escape_time ooo3 $args
rm opti3_escape_time

# ============================================================================ #
#                                                                              #
#                                      O1                                      #
#                                                                              #
# ============================================================================ #
make CC="gcc" CFLAGS="-g -pg -O1"

./opti3_escape_time ooo3 $args
rm opti3_escape_time

# ============================================================================ #
#                                                                              #
#                                      O2                                      #
#                                                                              #
# ============================================================================ #
make CC="gcc" CFLAGS="-g -pg -O2"

./opti3_escape_time ooo3 $args
rm opti3_escape_time

# ============================================================================ #
#                                                                              #
#                                      O3                                      #
#                                                                              #
# ============================================================================ #
make CC="gcc" CFLAGS="-g -pg -O3"

./opti3_escape_time ooo3 $args
rm opti3_escape_time

# ============================================================================ #
#                                                                              #
#                             O3 + loop vectorizer                             #
#                                                                              #
# ============================================================================ #
make CC="gcc" CFLAGS="-g -pg -O3 -ftree-vectorize"

./opti3_escape_time ooo3 $args

