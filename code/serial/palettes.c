#include "palettes.h"
#include <math.h>


/* -------------------------------------------------------------------------- */
void simple_colormap(int offset, int maxiter, int **iters,
                     int *rp, int *gp, int *bp)
{
  int v = (*iters)[offset] % maxiter;

//  *rp = floor(v/6 % 256);   // Red channel
  *rp = floor(v*6 % 256);   // Red channel
  *gp = floor(v*3/2 % 256); // Green channel
  *bp = floor(v*2 % 256);   // Blue channel
}

/* -------------------------------------------------------------------------- */
void sin_colormap(int offset, int maxiter, float **iters,
                  int *rp, int *gp, int *bp)
{
  float continuous_index = fmod((*iters)[offset], maxiter);

  *rp = sin(0.016 * continuous_index + 4) * 230 + 25; // Red channel
  *gp = sin(0.013 * continuous_index + 2) * 230 + 25; // Green channel
  *bp = sin(0.010 * continuous_index + 1) * 230 + 25; // Blue channel
}

/* -------------------------------------------------------------------------- */
void smooth_colormap(int offset, int maxiter, int **iters,
                     double **res, double **ims, int *rp, int *gp, int *bp)
{
  int iter = (*iters)[offset];

  if(iter == maxiter) {
    /* black */
    *rp = 0; // Red channel
    *gp = 0; // Green channel
    *bp = 0; // Blue channel
  }
  else {
    double re = (*res)[offset];
    double im = (*ims)[offset];
    double z = sqrt(re * re + im * im);
    int brightness = 256.*log2(1.75+iter-log2(log2(z))) / log2((double)maxiter);

    *rp = brightness; // Red channel
    *gp = brightness; // Green channel
    *bp = 255; // Blue channel
  }
}

void smooth_fast_colormap(int offset, int maxiter, int **iters, double **res2,
                          double **ims2, int *rp, int *gp, int *bp,
                          double cst_inside, double denominator)
{
  int iter = (*iters)[offset];

  if(iter == maxiter) {
    /* black */
    *rp = 0; // Red channel
    *gp = 0; // Green channel
    *bp = 0; // Blue channel
  }
  else {
    double re2 = (*res2)[offset];
    double im2 = (*ims2)[offset];
    int brightness = 256.*log2(cst_inside+iter-log2(log2(re2+im2)))/denominator;

    *rp = brightness; // Red channel
    *gp = brightness; // Green channel
    *bp = 255; // Blue channel
  }
}
