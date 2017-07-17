#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/******************************************************************************/
#define BLOCK_SIZE_X 256
#define BLOCK_SIZE_Y 1

#define PROCS_PER_NODE 2

__constant__ int c_maxiter;
__constant__ float c_xmin;
__constant__ float c_ymin;
__constant__ float c_x_step;
__constant__ float c_y_step;
__constant__ int c_N;
__constant__ int c_width;
__constant__ int c_rowsize;
__constant__ int c_mpi_rowoffset;

/******************************************************************************/

__device__ void d_smooth_fast_element_colormap(int iter, float re2, float im2,
                                               int *rp, int *gp, int *bp)
{
  if(iter == c_maxiter) {
    /* black */
    *rp = 0; // Red channel
    *gp = 0; // Green channel
    *bp = 0; // Blue channel
  }
  else {
    int brightness = 256.*log2(1.75-log2(0.5)+iter-log2(log2(re2+im2)))/log2((float)c_maxiter);

    *rp = brightness; // Red channel
    *gp = brightness; // Green channel
    *bp = 255; // Blue channel
  }
}


__device__ void in_cardioid_or_period2_bulb(int *iterp, float x, float y)
{
  float xdiff = x - 0.25;
  float y2 = y * y;
  float q = xdiff*xdiff + y2;

  // Is the point in the cardioid?
  if (q * (q + xdiff) < 0.25*y2) {
    *iterp = c_maxiter;
  }
  else if ((x+1.)*(x+1.) + y2 < 0.0625) { // Is the point in the period-2 bulb?
    *iterp = c_maxiter;
  }

}


__global__ void compute_escape_time(char *img)
{
  int offset = gridDim.x*blockDim.x*threadIdx.y + blockIdx.x*blockDim.x+threadIdx.x;
  int i = offset / c_width;
  int j = offset - i * c_width;
  int global_offset = c_mpi_rowoffset*c_width + offset;
  int global_i = global_offset / c_width;
  int global_j = global_offset - global_i * c_width;
  int iteration = 0;
  float c_re = c_xmin + c_x_step/2 + global_j*c_x_step;
  float c_im = c_ymin + c_y_step/2 + global_i*c_y_step;
  float zn_re = 0.;
  float zn_im = 0.;
  float tmp_re;
  float re2 = 0.;
  float im2 = 0.;
  int bailout_radius2 = 2*2;
  int r, g, b;

  if (global_offset < c_N) {
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

    d_smooth_fast_element_colormap(iteration, re2, im2, &r, &g, &b);

    offset = c_rowsize * i + 3 * j; // offset in the image array
    img[offset++] = b;
    img[offset++] = g;
    img[offset] = r;
  }
}

/******************************************************************************/

extern "C" void kernel_wrapper(char *h_img, int d_img_size, int MAX_ITER,
                               float X_MIN, float Y_MIN, float h_x_step,
                               float h_y_step, int N,
                               int N_local, int mpi_row_offset, int prank,
                               int WIDTH, int row_size)
{

  dim3 block_size, grid_size;
  char *d_img;
  int dev_id = prank % PROCS_PER_NODE;

  // Create the grid of blocks of threads
  block_size.x = BLOCK_SIZE_X; block_size.y = BLOCK_SIZE_Y;
  grid_size.x = N_local / (block_size.x*block_size.y) + (N_local%(block_size.x*block_size.y) == 0? 0 : 1);

  cudaSetDevice(dev_id);
  cudaMalloc((void **)&d_img, d_img_size);
  cudaMemset(d_img, 0, d_img_size);

  // Copy memory to constant memory in the device
  cudaMemcpyToSymbol(c_maxiter, &MAX_ITER, sizeof(int));
  cudaMemcpyToSymbol(c_xmin, &X_MIN, sizeof(float));
  cudaMemcpyToSymbol(c_ymin, &Y_MIN, sizeof(float));
  cudaMemcpyToSymbol(c_x_step, &h_x_step, sizeof(float));
  cudaMemcpyToSymbol(c_y_step, &h_y_step, sizeof(float));
  cudaMemcpyToSymbol(c_N, &N, sizeof(int));
  cudaMemcpyToSymbol(c_width, &WIDTH, sizeof(int));
  cudaMemcpyToSymbol(c_rowsize, &row_size, sizeof(int));
  cudaMemcpyToSymbol(c_mpi_rowoffset, &mpi_row_offset, sizeof(int));

  // Call the kernel to execute on the gpu
  compute_escape_time<<<grid_size, block_size>>>(d_img);

  // Copy the results back
  cudaMemcpy(h_img, d_img, d_img_size, cudaMemcpyDeviceToHost);

  cudaFree(d_img);
}
