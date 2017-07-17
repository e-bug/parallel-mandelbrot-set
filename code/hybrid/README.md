# Hybrid (CUDA+MPI) implementations

**Note:** before submitting any scripts or connecting into a GPU node, make sure you load the CUDA and MPI profile used in compilation: `source /ssoft/spack/bin/slmodules.sh -s x86_E5v2_Mellanox_GPU`

---

You can use the Makefile to build the executables; they will be compiled with the *DEBUG* flag.

- `plots/`: folder in which the produced images are stored.


## Source files
- `escape_time_5_1gpu_per_proc.cuh`: CUDA header for `escape_time_5_1gpu_per_proc.cu`.
- `escape_time_5_1gpu_per_proc.cu`: CUDA part for the implementation running on 2 processors and 2 GPUs (one per processor).
- `opti5_escape_time_1gpu_per_proc.c`: C (MPI) part for the implementation running on 2 processors and 2 GPUs (one per processor) where image parts are sent to rank 0 with non-blocking communications.
- `opti5-IO_escape_time_1gpu_per_proc.c`: C (MPI) part for the implementation running on 2 processors and 2 GPUs (one per processor) exploiting MPI parallel writing.
<br>
- `escape_time_5_2gpu_per_proc.cuh`: CUDA header for `escape_time_5_2gpu_per_proc.cu`.
- `escape_time_5_2gpu_per_proc.cu`: CUDA part for the implementation running on 2 processors and 4 GPUs (two per processor).
- `opti5_escape_time_2gpu_per_proc.c`: C (MPI) part for the implementation running on 2 processors and 4 GPUs (two per processor) where image parts are sent to rank 0 with non-blocking communications.
- `opti5-IO_escape_time_2gpu_per_proc.c`: C (MPI) part for the implementation running on 2 processors and 4 GPUs (two per processor) exploiting MPI parallel writing.


## Scripts
**Warning:** some GPU nodes on Deneb return the following error when using sbatch: `make: nvcc: Command not found`. 
If that's the case, use salloc instead: `salloc --partition=gpu --qos=gpu_free --nodes=1 --gres=gpu:2` or `salloc --partition=gpu --qos=gpu_free --nodes=1 --gres=gpu:4`.

- `submit_1gpu_per_proc_script.sh`: Example script to submit hybrid executables running on 2 processors and 2 GPUs on a GPU node. Launch it with: `sbatch submit_1gpu_per_proc_script.sh`.

- `submit_2gpu_per_proc_script.sh`: Example script to submit hybrid executables running on 2 processors and 4 GPUs  on a GPU node. Launch it with: `sbatch submit_2gpu_per_proc_script.sh`.

- `submit_scripts.sh`: Example script submitting all the hybrid implementations to produce small 1920x1080-pixel images. Launch it with: `sbatch submit_scripts.sh`.