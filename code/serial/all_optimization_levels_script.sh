#!/bin/bash
#SBATCH --reservation phpc2017
#SBATCH --account phpc2017
#SBATCH -N 1
#SBATCH -n 1


# For each optimization level and for each compiler (gcc and intel): 
# 1. create all executables with the -g -pg flags (for profiling)
# 2. run each executable producing an image with the optimization level and
#    the compiler used (e.g. _O0gcc means that gcc and -O0 were used)
# 3. rename the special file gmon.out produced by each executable and the 
#    executable to reflect which compier and optimization levels were used

args="2560 1440 10000"
# ============================================================================ #
#                                                                              #
#                                      O0                                      #
#                                                                              #
# ============================================================================ #
echo -e "O0"

# ============================================================================ #
#                                     gcc                                      #
# ============================================================================ #
module load gcc
make CC="gcc" CFLAGS="-g -pg -O0"

./naive_escape_time naive_O0gcc $args
mv gmon.out naive_O0gcc.out
mv naive_escape_time naive_escape_time_O0gcc

./naive_smooth_escape_time naivesmooth_O0gcc $args
mv gmon.out naive_smooth_O0gcc.out
mv naive_smooth_escape_time naive_smooth_escape_time_O0gcc

./opti1_escape_time opti1_O0gcc $args
mv gmon.out opti1_O0gcc.out
mv opti1_escape_time opti1_escape_time_O0gcc

./opti2_escape_time opti2_O0gcc $args
mv gmon.out opti2_O0gcc.out
mv opti2_escape_time opti2_escape_time_O0gcc

./opti3_escape_time opti3_O0gcc $args
mv gmon.out opti3_O0gcc.out
mv opti3_escape_time opti3_escape_time_O0gcc

# ============================================================================ #
#                                    intel                                     #
# ============================================================================ #
module load intel
make CC="icpc" CFLAGS="-g -pg -O0"

./naive_escape_time naive_O0intel $args
mv gmon.out naive_O0intel.out
mv naive_escape_time naive_escape_time_O0intel

./naive_smooth_escape_time naivesmooth_O0intel $args
mv gmon.out naive_smooth_O0intel.out
mv naive_smooth_escape_time naive_smooth_escape_time_O0intel

./opti1_escape_time opti1_O0intel $args
mv gmon.out opti1_O0intel.out
mv opti1_escape_time opti1_escape_time_O0intel

./opti2_escape_time opti2_O0intel $args
mv gmon.out opti2_O0intel.out
mv opti2_escape_time opti2_escape_time_O0intel

./opti3_escape_time opti3_O0intel $args
mv gmon.out opti3_O0intel.out
mv opti3_escape_time opti3_escape_time_O0intel


echo -e "\n"


# ============================================================================ #
#                                                                              #
#                                      O1                                      #
#                                                                              #
# ============================================================================ #
echo -e "O1"

# ============================================================================ #
#                                     gcc                                      #
# ============================================================================ #
module load gcc
make CC="gcc" CFLAGS="-g -pg -O1"

./naive_escape_time naive_O0gcc $args
mv gmon.out naive_O1gcc.out
mv naive_escape_time naive_escape_time_O1gcc

./naive_smooth_escape_time naivesmooth_O1gcc $args
mv gmon.out naive_smooth_O1gcc.out
mv naive_smooth_escape_time naive_smooth_escape_time_O1gcc

./opti1_escape_time opti1_O1gcc $args
mv gmon.out opti1_O1gcc.out
mv opti1_escape_time opti1_escape_time_O1gcc

./opti2_escape_time opti2_O1gcc $args
mv gmon.out opti2_O1gcc.out
mv opti2_escape_time opti2_escape_time_O1gcc

./opti3_escape_time opti3_O1gcc $args
mv gmon.out opti3_O1gcc.out
mv opti3_escape_time opti3_escape_time_O1gcc

# ============================================================================ #
#                                    intel                                     #
# ============================================================================ #
module load intel
make CC="icpc" CFLAGS="-g -pg -O1"

./naive_escape_time naive_O1intel $args
mv gmon.out naive_O1intel.out
mv naive_escape_time naive_escape_time_O1intel

./naive_smooth_escape_time naivesmooth_O1intel $args
mv gmon.out naive_smooth_O1intel.out
mv naive_smooth_escape_time naive_smooth_escape_time_O1intel

./opti1_escape_time opti1_O1intel $args
mv gmon.out opti1_O1intel.out
mv opti1_escape_time opti1_escape_time_O1intel

./opti2_escape_time opti2_O1intel $args
mv gmon.out opti2_O1intel.out
mv opti2_escape_time opti2_escape_time_O1intel

./opti3_escape_time opti3_O1intel $args
mv gmon.out opti3_O1intel.out
mv opti3_escape_time opti3_escape_time_O1intel


echo -e "\n"


# ============================================================================ #
#                                                                              #
#                                      O2                                      #
#                                                                              #
# ============================================================================ #
echo -e "O2"

# ============================================================================ #
#                                     gcc                                      #
# ============================================================================ #
module load gcc
make CC="gcc" CFLAGS="-g -pg -O2"

./naive_escape_time naive_O2gcc $args
mv gmon.out naive_O2gcc.out
mv naive_escape_time naive_escape_time_O2gcc

./naive_smooth_escape_time naivesmooth_O2gcc $args
mv gmon.out naive_smooth_O2gcc.out
mv naive_smooth_escape_time naive_smooth_escape_time_O2gcc

./opti1_escape_time opti1_O2gcc $args
mv gmon.out opti1_O2gcc.out
mv opti1_escape_time opti1_escape_time_O2gcc

./opti2_escape_time opti2_O2gcc $args
mv gmon.out opti2_O2gcc.out
mv opti2_escape_time opti2_escape_time_O2gcc

./opti3_escape_time opti3_O2gcc $args
mv gmon.out opti3_O2gcc.out
mv opti3_escape_time opti3_escape_time_O2gcc

# ============================================================================ #
#                                    intel                                     #
# ============================================================================ #
module load intel
make CC="icpc" CFLAGS="-g -pg -O2"

./naive_escape_time naive_O2intel $args
mv gmon.out naive_O2intel.out
mv naive_escape_time naive_escape_time_O2intel

./naive_smooth_escape_time naivesmooth_O2intel $args
mv gmon.out naive_smooth_O2intel.out
mv naive_smooth_escape_time naive_smooth_escape_time_O2intel

./opti1_escape_time opti1_O2intel $args
mv gmon.out opti1_O2intel.out
mv opti1_escape_time opti1_escape_time_O2intel

./opti2_escape_time opti2_O2intel $args
mv gmon.out opti2_O2intel.out
mv opti2_escape_time opti2_escape_time_O2intel

./opti3_escape_time opti3_O2intel $args
mv gmon.out opti3_O2intel.out
mv opti3_escape_time opti3_escape_time_O2intel


echo -e "\n"


# ============================================================================ #
#                                                                              #
#                                      O3                                      #
#                                                                              #
# ============================================================================ #
echo -e "O3"

# ============================================================================ #
#                                     gcc                                      #
# ============================================================================ #
module load gcc
make CC="gcc" CFLAGS="-g -pg -O3"

./naive_escape_time naive_O3gcc $args
mv gmon.out naive_O3gcc.out
mv naive_escape_time naive_escape_time_O3gcc

./naive_smooth_escape_time naivesmooth_O3gcc $args
mv gmon.out naive_smooth_O3gcc.out
mv naive_smooth_escape_time naive_smooth_escape_time_O3gcc

./opti1_escape_time opti1_O3gcc $args
mv gmon.out opti1_O3gcc.out
mv opti1_escape_time opti1_escape_time_O3gcc

./opti2_escape_time opti2_O3gcc $args
mv gmon.out opti2_O3gcc.out
mv opti2_escape_time opti2_escape_time_O3gcc

./opti3_escape_time opti3_O3gcc $args
mv gmon.out opti3_O3gcc.out
mv opti3_escape_time opti3_escape_time_O3gcc

# ============================================================================ #
#                                    intel                                     #
# ============================================================================ #
module load intel
make CC="icpc" CFLAGS="-g -pg -O3"

./naive_escape_time naive_O3intel $args
mv gmon.out naive_O3intel.out
mv naive_escape_time naive_escape_time_O3intel

./naive_smooth_escape_time naivesmooth_O3intel $args
mv gmon.out naive_smooth_O3intel.out
mv naive_smooth_escape_time naive_smooth_escape_time_O3intel

./opti1_escape_time opti1_O3intel $args
mv gmon.out opti1_O3intel.out
mv opti1_escape_time opti1_escape_time_O3intel

./opti2_escape_time opti2_O3intel $args
mv gmon.out opti2_O3intel.out
mv opti2_escape_time opti2_escape_time_O3intel

./opti3_escape_time opti3_O3intel $args
mv gmon.out opti3_O3intel.out
mv opti3_escape_time opti3_escape_time_O3intel


echo -e "\n"


# ============================================================================ #
#                                                                              #
#                             O3 + loop vectorizer                             #
#                                                                              #
# ============================================================================ #
echo -e "O3V"

# ============================================================================ #
#                                     gcc                                      #
# ============================================================================ #
module load gcc
make CC="gcc" CFLAGS="-g -pg -O3 -ftree-vectorize"

./naive_escape_time naive_O3Vgcc $args
mv gmon.out naive_O3Vgcc.out
mv naive_escape_time naive_escape_time_O3Vgcc

./naive_smooth_escape_time naivesmooth_O3Vgcc $args
mv gmon.out naive_smooth_O3Vgcc.out
mv naive_smooth_escape_time naive_smooth_escape_time_O3Vgcc

./opti1_escape_time opti1_O3Vgcc $args
mv gmon.out opti1_O3Vgcc.out
mv opti1_escape_time opti1_escape_time_O3Vgcc

./opti2_escape_time opti2_O3Vgcc $args
mv gmon.out opti2_O3Vgcc.out
mv opti2_escape_time opti2_escape_time_O3Vgcc

./opti3_escape_time opti3_O3Vgcc $args
mv gmon.out opti3_O3Vgcc.out
mv opti3_escape_time opti3_escape_time_O3Vgcc

# ============================================================================ #
#                                    intel                                     #
# ============================================================================ #
module load intel
make CC="icpc" CFLAGS="-g -pg -O3 -xHost"

./naive_escape_time naive_O3Vintel $args
mv gmon.out naive_O3Vintel.out
mv naive_escape_time naive_escape_time_O3Vintel

./naive_smooth_escape_time naivesmooth_O3Vintel $args
mv gmon.out naive_smooth_O3Vintel.out
mv naive_smooth_escape_time naive_smooth_escape_time_O3Vintel

./opti1_escape_time opti1_O3Vintel $args
mv gmon.out opti1_O3Vintel.out
mv opti1_escape_time opti1_escape_time_O3Vintel

./opti2_escape_time opti2_O3Vintel $args
mv gmon.out opti2_O3Vintel.out
mv opti2_escape_time opti2_escape_time_O3Vintel

./opti3_escape_time opti3_O3Vintel $args
mv gmon.out opti3_O3Vintel.out
mv opti3_escape_time opti3_escape_time_O3Vintel


echo -e "\n"
