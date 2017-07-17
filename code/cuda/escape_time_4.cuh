#ifndef _ESCAPE_TIME_4_H_
#define _ESCAPE_TIME_4_H_


__device__ void d_smooth_fast_element_colormap(int iter, double re2, double im2,
                                               int *rp, int *gp, int *bp);
__device__ void in_cardioid_or_period2_bulb(int *iterp, double x, double y);
__global__ void compute_escape_time(char *img);

void kernel_wrapper(char *h_img, int d_img_size, int MAX_ITER, double X_MIN,
                    double Y_MIN, double h_x_step, double h_y_step,
                    int N, int WIDTH, int row_size);

#endif // _ESCAPE_TIME_4_H_
