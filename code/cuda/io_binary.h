#ifndef IO_BINARY_H
#define IO_BINARY_H

int draw_mandelbrot(int m, int n, int **iters, int maxiter, char *filename);
int draw_smooth_mandelbrot(int m, int n, int **iters, double **res,
                           double **ims, int maxiter, char *filename);
int draw_symmetric_mandelbrot(int m, int n, int **iters, double **res,
                          double **ims, int maxiter, char *filename);
int draw_fast_mandelbrot(int m, int n, int **iters, double **res2,
                         double **ims2, int maxiter, char *filename);

#endif
