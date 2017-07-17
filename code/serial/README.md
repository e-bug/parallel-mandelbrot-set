# Serial implementations

You can use the Makefile to build the executables; they will be compiled with the *DEBUG* flag.

- `plots/`: folder in which the produced images are stored.


## Source files
- `io_binary.h`: C header file for `io_binary.c`.
- `io_binary.c`: C file with all the used plotting functions.
<br>
- `palettes.h`: C header file for `palettes.c`.
- `palettes.c`: C file with different colormaps for the produced image.
<br>
- `naive_escape_time.c`: C file for the naive implementation producing red-hue images. 
- `naive_smooth_escape_time.c`: C file for the naive implementation producing smooth, blue-hue images. 
- `opti1_escape_time.c`: C file for the implementation exploiting the symmetry with respect to the x axis and producing smooth, blue-hue images.
- `opti2_escape_time.c`: C file for the implementation also skipping pixels in the cardioid and period-2 bulb and producing smooth, blue-hue images.
- `opti3_escape_time.c`: C file for the implementation with finer-grained optimizations and producing smooth, blue-hue images.


## Scripts
- `all_optimization_levels_script.sh`: Bash script that compiles each implementation with the "-g -pg" flags for each optimization level and with each compiler (gcc and intel).
- `opti3_gcc_optimization_levels_script.sh`: Bash script that compiles the *opti3* implementation with the "-g -pg" flags and runs it locally for each optimization level with the gcc compiler.
<br>
- `times_naive.sh`: Executes `naive_smooth_escape_time` for different image sizes and max_iteration values.
- `times_opti1.sh`: Executes `opti1_escape_time` for different image sizes and max_iteration values.
- `times_opti2.sh`: Executes `opti2_escape_time` for different image sizes and max_iteration values.
- `times_opti3.sh`: Executes `opti3_escape_time` for different image sizes and max_iteration values.