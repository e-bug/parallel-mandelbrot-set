#ifndef _ESCAPE_TIME_5_2GPU_PER_PROC_H_
#define _ESCAPE_TIME_5_2GPU_PER_PROC_H_


__device__ void d_smooth_fast_element_colormap(int iter, float re2, float im2,
                                               int *rp, int *gp, int *bp);
__device__ void in_cardioid_or_period2_bulb(int *iterp, float x, float y);
__global__ void compute_escape_time(char *img);

void kernel_wrapper(char *h_img, int d_img_size, int MAX_ITER, float X_MIN,
                    float Y_MIN, float h_x_step, float h_y_step,
                    int N, int N_local, int prank, int WIDTH, int row_size);

#endif // _ESCAPE_TIME_5_2GPU_PER_PROC_H_
