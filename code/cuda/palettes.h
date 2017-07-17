#ifndef PALETTES_H
#define PALETTES_H


void simple_colormap(int offset, int maxiter, int **iters,
                     int *rp, int *gp, int *bp);

void sin_colormap(int offset, int maxiter, float **iters,
                  int *rp, int *gp, int *bp);

void smooth_colormap(int offset, int maxiter, int **iters,
                     double **res, double **ims, int *rp, int *gp, int *bp);
void smooth_fast_colormap(int offset, int maxiter, int **iters, double **res2,
                          double **ims2, int *rp, int *gp, int *bp,
                          double cst_inside, double denominator);
void smooth_fast_element_colormap(int iter, int maxiter, double re2,
                                  double im2, int *rp, int *gp, int *bp);

#endif
