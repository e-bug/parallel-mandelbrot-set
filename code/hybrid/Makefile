NVCC=nvcc 
CC=mpicc
LD=${CC}
NLD=${NVCC}

NVCCFLAGS=-ldl -lrt -O3 -arch=sm_30
CFLAGS=-O3 -ftree-vectorize -DDEBUG
LDFLAGS+=-lm
MPI_ROOT=/ssoft/spack/lafnetscha/opt/spack/x86_E5v2_Mellanox_GPU/gcc-5.3.0/mvapich2-2.2b-tvty7u62pjhjto6yeryc6xf52lxagcrp
NVCCMPIFLAGS=-I ${MPI_ROOT}/include -L ${MPI_ROOT}/lib -l mpi

all: opti5_escape_time_1gpu_per_proc opti5-IO_escape_time_1gpu_per_proc \
     opti5_escape_time_2gpu_per_proc opti5-IO_escape_time_2gpu_per_proc \
     clean

opti5_escape_time_1gpu_per_proc:
	$(NLD) -c $(NVCCFLAGS) $(LDFLAGS) escape_time_5_1gpu_per_proc.cu
	$(LD) -c $(CFLAGS) $(LDFLAGS) opti5_escape_time_1gpu_per_proc.c
	$(NLD) $(NVCCMPIFLAGS) escape_time_5_1gpu_per_proc.o opti5_escape_time_1gpu_per_proc.o -o $@

opti5-IO_escape_time_1gpu_per_proc:
	$(NLD) -c $(NVCCFLAGS) $(LDFLAGS) escape_time_5_1gpu_per_proc.cu
	$(LD) -c $(CFLAGS) $(LDFLAGS) opti5-IO_escape_time_1gpu_per_proc.c
	$(NLD) $(NVCCMPIFLAGS) escape_time_5_1gpu_per_proc.o opti5-IO_escape_time_1gpu_per_proc.o -o $@

opti5_escape_time_2gpu_per_proc:
	$(NLD) -c $(NVCCFLAGS) $(LDFLAGS) escape_time_5_2gpu_per_proc.cu
	$(LD) -c $(CFLAGS) $(LDFLAGS) opti5_escape_time_2gpu_per_proc.c
	$(NLD) $(NVCCMPIFLAGS) escape_time_5_2gpu_per_proc.o opti5_escape_time_2gpu_per_proc.o -o $@

opti5-IO_escape_time_2gpu_per_proc:
	$(NLD) -c $(NVCCFLAGS) $(LDFLAGS) escape_time_5_2gpu_per_proc.cu
	$(LD) -c $(CFLAGS) $(LDFLAGS) opti5-IO_escape_time_2gpu_per_proc.c
	$(NLD) $(NVCCMPIFLAGS) escape_time_5_2gpu_per_proc.o opti5-IO_escape_time_2gpu_per_proc.o -o $@

clean:
	rm *.o
