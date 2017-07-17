#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" {
  #include "io_binary.h"
}

#ifdef DEBUG
  #include <time.h>
#endif

/******************************************************************************/
static const double X_MAX = 1.0;
static const double X_MIN = -2.5;
static const double Y_MAX = 1.0;
static const double Y_MIN = -1.0;

__constant__ int c_maxiter;
__constant__ double c_xmin;
__constant__ double c_ymin;
__constant__ double c_x_step;
__constant__ double c_y_step;
__constant__ int c_N;
__constant__ int c_width;

/******************************************************************************/
void allocate_int_grids(int **h_vec, int **d_vec, int n_bytes);
void allocate_double_grids(double **h_vec, double **d_vec, int n_bytes);
void deallocate_int_grids(int **h_vec, int **d_vec);
void deallocate_double_grids(double **h_vec, double **d_vec);

__device__ void in_cardioid_or_period2_bulb(int *iterp, double x, double y);
__global__ void compute_escape_time(int *iters, double *res, double *ims);

/******************************************************************************/

int main(int argc, char *argv[]) {

  char *filename;
  int WIDTH = 1920;
  int HEIGHT = 1080;
  int MAX_ITER = 1000;

  int *h_iters;
  double *h_res2, *h_ims2;
  double h_x_step, h_y_step;

  int N, N_BYTES_INT, N_BYTES_DOUBLE;
  int *d_iters;
  double *d_res2, *d_ims2;
  dim3 block_size, grid_size;

  /****************************************************************************/
  /*                            GET USER PARAMETERS                           */
  /****************************************************************************/
  if (argc == 2) {
    // Use default WIDTH, HEIGHT and MAX_ITER
    filename = argv[1];
  }
  else if (argc == 4) {
    // Use default MAX_ITER
    filename = argv[1];
    WIDTH = atoi(argv[2]);
    HEIGHT = atoi(argv[3]);
    if (HEIGHT % 2 != 0) {
      fprintf(stderr, "Incorrect usage: HEIGHT must be even"
                      "for the optimized version\n");
      exit(1);
    }
  }
  else if (argc == 5) {
    filename = argv[1];
    WIDTH = atoi(argv[2]);
    HEIGHT = atoi(argv[3]);
    MAX_ITER = atoi(argv[4]);
    if (HEIGHT % 2 != 0) {
      fprintf(stderr, "Incorrect usage: HEIGHT must be even"
                      "for the optimized version\n");
      exit(1);
    }
  }
  else {
    fprintf(stderr, "Incorrect usage. Ways to run the application:\n");
    fprintf(stderr, "%s <output_filename>\n", argv[0]);
    fprintf(stderr, "%s <output_filename> <WIDTH> <HEIGHT>\n", argv[0]);
    fprintf(stderr, "%s <output_filename> <WIDTH> <HEIGHT> <MAX_ITER>\n\n",
            argv[0]);
    exit(1);
  }

  /****************************************************************************/
  /*                              INITIALIZATION                              */
  /****************************************************************************/
  #ifdef DEBUG
    clock_t begin = clock();
  #endif

  h_x_step = (X_MAX - X_MIN) / WIDTH;
  h_y_step = (Y_MAX - Y_MIN) / HEIGHT;

  HEIGHT /= 2; // Symmetry with respect to the x axis

  N = WIDTH * HEIGHT;
  N_BYTES_INT = N * sizeof(int);
  N_BYTES_DOUBLE = N * sizeof(double);

  // Create the grid of blocks of threads
  block_size.x = 512;
  grid_size.x = N / block_size.x + (N%block_size.x == 0? 0 : 1);


  // Allocate memory on the host and on the device
  allocate_int_grids(&h_iters, &d_iters, N_BYTES_INT);
  allocate_double_grids(&h_res2, &d_res2, N_BYTES_DOUBLE);
  allocate_double_grids(&h_ims2, &d_ims2, N_BYTES_DOUBLE);

  // Copy memory to constant memory in the device
  cudaMemcpyToSymbol(c_maxiter, &MAX_ITER, sizeof(int));
  cudaMemcpyToSymbol(c_xmin, &X_MIN, sizeof(double));
  cudaMemcpyToSymbol(c_ymin, &Y_MIN, sizeof(double));
  cudaMemcpyToSymbol(c_x_step, &h_x_step, sizeof(double));
  cudaMemcpyToSymbol(c_y_step, &h_y_step, sizeof(double));
  cudaMemcpyToSymbol(c_N, &N, sizeof(int));
  cudaMemcpyToSymbol(c_width, &WIDTH, sizeof(int));

  /****************************************************************************/
  /*                           EVALUATE ESCAPE TIMES                          */
  /****************************************************************************/
  // Call the kernel to execute on the gpu
  compute_escape_time<<<grid_size, block_size>>>(d_iters, d_res2, d_ims2);

  // Copy the results back
  cudaMemcpy(h_iters, d_iters, N_BYTES_INT, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_res2, d_res2, N_BYTES_DOUBLE, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ims2, d_ims2, N_BYTES_DOUBLE, cudaMemcpyDeviceToHost);

  /****************************************************************************/
  /*                            DRAW MANDELBROT SET                           */
  /****************************************************************************/
  draw_fast_mandelbrot(HEIGHT,WIDTH,&h_iters,&h_res2,&h_ims2,MAX_ITER,filename);

  /****************************************************************************/
  /*                              FREE RESOURCES                              */
  /****************************************************************************/
  deallocate_int_grids(&h_iters, &d_iters);
  deallocate_double_grids(&h_res2, &d_res2);
  deallocate_double_grids(&h_ims2, &d_ims2);

  #ifdef DEBUG
    fprintf(stdout, "Execution time = %f [s]\n",
            (double)(clock() - begin) / CLOCKS_PER_SEC);
  #endif

  return 0;
}


/******************************************************************************/
void allocate_int_grids(int **h_vec, int **d_vec, int n_bytes) {
  *h_vec = (int *) malloc(n_bytes);
  cudaMalloc((void **)d_vec, n_bytes);
}

void allocate_double_grids(double **h_vec, double **d_vec, int n_bytes) {
  *h_vec = (double *) malloc(n_bytes);
  cudaMalloc((void **)d_vec, n_bytes);
}

void deallocate_int_grids(int **h_vec, int **d_vec) {
  free(*h_vec);
  cudaFree(*d_vec);
}

void deallocate_double_grids(double **h_vec, double **d_vec) {
  free(*h_vec);
  cudaFree(*d_vec);
}


__device__ void in_cardioid_or_period2_bulb(int *iterp, double x, double y)
{
  double xdiff = x - 0.25;
  double y2 = y * y;
  double q = xdiff*xdiff + y2;

  // Is the point in the cardioid?
  if (q * (q + xdiff) < 0.25*y2) {
    *iterp = c_maxiter;
  }
  else if ((x+1.)*(x+1.) + y2 < 0.0625) { // Is the point in the period-2 bulb?
    *iterp = c_maxiter;
  }

}


__global__ void compute_escape_time(int *iters, double *res2, double *ims2)
{
  int offset = blockIdx.x*blockDim.x + threadIdx.x;
  int i = offset / c_width;
  int j = offset - i * c_width;
  int iteration = 0;
  double c_re = c_xmin + c_x_step/2 + j*c_x_step;
  double c_im = c_ymin + c_y_step/2 + i*c_y_step;
  double zn_re = 0.;
  double zn_im = 0.;
  double tmp_re;
  double re2 = 0.;
  double im2 = 0.;
  int bailout_radius2 = 2*2;

  if (offset < c_N) {
    // Check if point is in cardioid or in period-2 bulb
    in_cardioid_or_period2_bulb(&iteration, c_re, c_im);

    while ((re2 + im2 < bailout_radius2) && (iteration < c_maxiter)) {
      tmp_re = re2 - im2 + c_re;
      zn_im = zn_re * zn_im;
      zn_im += zn_im; // Multiply by two
      zn_im += c_im;
      zn_re = tmp_re;

      re2 = zn_re * zn_re;
      im2 = zn_im * zn_im;
      iteration++;
    }

    iters[offset] = iteration;
    res2[offset] = re2;
    ims2[offset] = im2;
  }
}
