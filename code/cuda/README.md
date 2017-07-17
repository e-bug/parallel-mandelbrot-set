# CUDA implementations

You can use the Makefile to build the executables; they will be compiled with the *DEBUG* flag.

- `plots/`: folder in which the produced images are stored.


## Source files
- `io_binary.h`: C header file for `io_binary.c`.
- `io_binary.c`: C file with all the used plotting functions.

- `palettes.h`: C header file for `palettes.c`.
- `palettes.c`: C file with different colormaps for the produced image.

- `opti3_escape_time.cu`: CUDA implementation of *opti3*, only parallelizing the escape time evaluations for each pixel.

- `escape_time_4.cuh`: CUDA header for `escape_time_4.cu`.
- `escape_time_4.cu`: CUDA part for the implementation also parallelizing the evaluation of the color for each pixel.
- `opti4_escape_time.c`: C part for the implementation also parallelizing the evaluation of the color for each pixel.

- `escape_time_5.cuh`: CUDA header for `escape_time_5.cu`.
- `escape_time_5.cu`: CUDA part for the single-precision implementation of *opti4*.
- `opti5_escape_time.c`: C part for the single-precision implementation of *opti4*.


## Scripts
**Warning:** some GPU nodes on Deneb return the following error when using sbatch: `make: nvcc: Command not found`. 
If that's the case, use salloc instead: `salloc --partition=gpu --qos=gpu_free --nodes=1 --gres=gpu:1`.

- `submit_script.sh`: Example script to submit CUDA executables on a GPU node. Launch it with: `sbatch submit_script.sh`.

- `nvprof_script.sh`: Example script making use of  *nvprof* to profile an application and to evaluate the achieved occupancy.
