# A High-Performance Implementation of the Mandelbrot Set

Repository for the project in Parallel and High-Performance Computing, EPFL 2016/2017. <br>
<hr />

A parallel implementation of the “escape time” algorithm to produce drawings of the Mandelbrot Set. <br> 
Application coded in C; parallel implementations in CUDA and CUDA+MPI.

![Alt text](presentation/figures/o3.bmp)


## Overview

The aim of this project is to obtain a parallal and high performance implementation to generate images of the Mandelbrot Set.

After optimizing the serial version, we move the workload to the GPU using CUDA and tune hyperparameters, like the block size, to increase the application's achieved occupancy and obtain a first optimized parallel version. <br>
Finally, we distribute the workload to different processors and multiple GPUs using MPI (including MPI-IO for parallel I/O).

The performance of these versions is then assessed in terms of Strong Scaling and Weak Scaling.


## Description of this repository

- `code/`: source code for the application, from serial to hybrid versions. Step-by-step optimizations are provided.
- `plots/`: plots used in the report and the IPython notebooks used to produce them.
- `presentation/`: slides shown during the final presentation.
- `proposal/`: submitted final report.
- `results/`: execution times of the different versions; used to generate the plots. Each value is the median of five measurements.
