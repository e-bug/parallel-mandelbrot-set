NVCC=nvcc 
CC=g++
LD=${CC}
NLD=${NVCC}

NVCCFLAGS=-ldl -lrt -O3 -arch=sm_30
CFLAGS=-O3 -ftree-vectorize -DDEBUG

LDFLAGS+=-lm


all: opti3_escape_time opti4_escape_time opti5_escape_time clean

opti3_escape_time: palettes.c io_binary.c opti3_escape_time.cu
	$(NLD) $(NVCCFLAGS) $? -g -pg -DDEBUG $(LDFLAGS) -o $@

opti4_escape_time:
	$(LD) -c $(CFLAGS) $(LDFLAGS) opti4_escape_time.c
	$(NLD) -c $(NVCCFLAGS) $(LDFLAGS) escape_time_4.cu
	$(NLD) escape_time_4.o opti4_escape_time.o -o $@

opti5_escape_time:
	$(LD) -c $(CFLAGS) $(LDFLAGS) opti5_escape_time.c
	$(NLD) -c $(NVCCFLAGS) $(LDFLAGS) escape_time_5.cu
	$(NLD) escape_time_5.o opti5_escape_time.o -o $@

clean:
	rm *.o
