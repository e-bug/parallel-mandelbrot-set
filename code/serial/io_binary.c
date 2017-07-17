#include "io_binary.h"
#include "palettes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ************************************************************************** */
/* ************************************************************************** */
/* ************************************************************************** */
int draw_mandelbrot(int m, int n, int **iters, int maxiter, char *filename) {

  FILE *fp;
  char prefix[] = "plots/";
  char suffix[] = ".bmp";

  int row_size, padding, filesize;
  char *img;

  int i, j, offset;
  int r, g, b;

  char bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
  char bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};


  strcat(prefix, filename);
  strcat(prefix, suffix);
  fp = fopen(prefix, "wb");
  if (fp == NULL) {
    fprintf(stderr, "Can't open output file %s!\n", prefix);
    exit(1);
  }

  /* The length of each line must be a multiple of 4 bytes */
  row_size = 3 * n;
  // if the file width (3*n) is not a multiple of 4 adds enough bytes to make it
  // a multiple of 4
  padding = (4 - (row_size) % 4) % 4;
  row_size += padding;

  filesize = 54 + (row_size)*m;

  img = (char *) malloc(row_size*m * sizeof(char));
  for (i = 0; i < row_size*m; ++i) img[i] = 0;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      offset = i * n + j;
      simple_colormap(offset, maxiter, iters, &r, &g, &b);

      img[row_size * i + 3 * j + 2] = r;
      img[row_size * i + 3 * j + 1] = g;
      img[row_size * i + 3 * j + 0] = b;

    }
  }

  bmpfileheader[2] = filesize;
  bmpfileheader[3] = filesize >> 8;
  bmpfileheader[4] = filesize >> 16;
  bmpfileheader[5] = filesize >> 24;

  bmpinfoheader[4]  = n;
  bmpinfoheader[5]  = n >> 8;
  bmpinfoheader[6]  = n >> 16;
  bmpinfoheader[7]  = n >> 24;
  bmpinfoheader[8]  = m;
  bmpinfoheader[9]  = m >> 8;
  bmpinfoheader[10] = m >> 16;
  bmpinfoheader[11] = m >> 24;
  bmpinfoheader[20] = (filesize - 54);
  bmpinfoheader[21] = (filesize - 54) >> 8;
  bmpinfoheader[22] = (filesize - 54) >> 16;
  bmpinfoheader[23] = (filesize - 54) >> 24;

  fwrite(bmpfileheader, 1, 14, fp);
  fwrite(bmpinfoheader, 1, 40, fp);

  fwrite(img, 1, m * row_size, fp);

  fclose(fp);
  return 0;
}

/* ************************************************************************** */
/* ************************************************************************** */
/* ************************************************************************** */
int draw_smooth_mandelbrot(int m, int n, int **iters, double **res,
                           double **ims, int maxiter, char *filename)
{
  FILE *fp;
  char prefix[] = "plots/";
  char suffix[] = ".bmp";

  int row_size, padding, filesize;
  char *img;

  int i, j, offset;
  int r, g, b;

  char bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
  char bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};


  strcat(prefix, filename);
  strcat(prefix, suffix);
  fp = fopen(prefix, "wb");
  if (fp == NULL) {
    fprintf(stderr, "Can't open output file %s!\n", prefix);
    exit(1);
  }

  /* The length of each line must be a multiple of 4 bytes */
  row_size = 3 * n;
  // if the file width (3*n) is not a multiple of 4 adds enough bytes to make it
  // a multiple of 4
  padding = (4 - (row_size) % 4) % 4;
  row_size += padding;

  filesize = 54 + (row_size)*m;

  img = (char *) malloc(row_size*m * sizeof(char));
  for (i = 0; i < row_size*m; ++i) img[i] = 0;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      offset = i * n + j;
      smooth_colormap(offset, maxiter, iters, res, ims, &r, &g, &b);

      img[row_size * i + 3 * j + 2] = r;
      img[row_size * i + 3 * j + 1] = g;
      img[row_size * i + 3 * j + 0] = b;

    }
  }

  bmpfileheader[2] = filesize;
  bmpfileheader[3] = filesize >> 8;
  bmpfileheader[4] = filesize >> 16;
  bmpfileheader[5] = filesize >> 24;

  bmpinfoheader[4]  = n;
  bmpinfoheader[5]  = n >> 8;
  bmpinfoheader[6]  = n >> 16;
  bmpinfoheader[7]  = n >> 24;
  bmpinfoheader[8]  = m;
  bmpinfoheader[9]  = m >> 8;
  bmpinfoheader[10] = m >> 16;
  bmpinfoheader[11] = m >> 24;
  bmpinfoheader[20] = (filesize - 54);
  bmpinfoheader[21] = (filesize - 54) >> 8;
  bmpinfoheader[22] = (filesize - 54) >> 16;
  bmpinfoheader[23] = (filesize - 54) >> 24;

  fwrite(bmpfileheader, 1, 14, fp);
  fwrite(bmpinfoheader, 1, 40, fp);

  fwrite(img, 1, m * row_size, fp);

  fclose(fp);
  return 0;
}

/* ************************************************************************** */
/* ************************************************************************** */
/* ************************************************************************** */
int draw_symmetric_mandelbrot(int m, int n, int **iters, double **res,
                              double **ims, int maxiter, char *filename)
{
  FILE *fp;
  char prefix[] = "plots/";
  char suffix[] = ".bmp";

  int M = m*2;
  int row_size, padding, filesize;
  char *img;

  int i, j, offset;
  int r, g, b;

  char bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
  char bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};


  strcat(prefix, filename);
  strcat(prefix, suffix);
  fp = fopen(prefix, "wb");
  if (fp == NULL) {
    fprintf(stderr, "Can't open output file %s!\n", prefix);
    exit(1);
  }

  /* The length of each line must be a multiple of 4 bytes */
  row_size = 3 * n;
  // if the file width (3*n) is not a multiple of 4 adds enough bytes to make it
  // a multiple of 4
  padding = (4 - (row_size) % 4) % 4;
  row_size += padding;

  filesize = 54 + (row_size)*M;

  img = (char *) malloc(row_size*M * sizeof(char));
  for (i = 0; i < row_size*M; ++i) img[i] = 0;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      offset = i * n + j;
      smooth_colormap(offset, maxiter, iters, res, ims, &r, &g, &b);

      img[row_size * i + 3 * j + 2] = r;
      img[row_size * i + 3 * j + 1] = g;
      img[row_size * i + 3 * j + 0] = b;

      // Symmetric draw
      img[row_size * (M-1-i) + 3 * j + 2] = r;
      img[row_size * (M-1-i) + 3 * j + 1] = g;
      img[row_size * (M-1-i) + 3 * j + 0] = b;

    }
  }

  bmpfileheader[2] = filesize;
  bmpfileheader[3] = filesize >> 8;
  bmpfileheader[4] = filesize >> 16;
  bmpfileheader[5] = filesize >> 24;

  bmpinfoheader[4]  = n;
  bmpinfoheader[5]  = n >> 8;
  bmpinfoheader[6]  = n >> 16;
  bmpinfoheader[7]  = n >> 24;
  bmpinfoheader[8]  = M;
  bmpinfoheader[9]  = M >> 8;
  bmpinfoheader[10] = M >> 16;
  bmpinfoheader[11] = M >> 24;
  bmpinfoheader[20] = (filesize - 54);
  bmpinfoheader[21] = (filesize - 54) >> 8;
  bmpinfoheader[22] = (filesize - 54) >> 16;
  bmpinfoheader[23] = (filesize - 54) >> 24;

  fwrite(bmpfileheader, 1, 14, fp);
  fwrite(bmpinfoheader, 1, 40, fp);

  fwrite(img, 1, M * row_size, fp);

  fclose(fp);
  return 0;
}

/* ************************************************************************** */
/* ************************************************************************** */
/* ************************************************************************** */
int draw_fast_mandelbrot(int m, int n, int **iters, double **res2,
                         double **ims2, int maxiter, char *filename)
{
  FILE *fp;
  char prefix[] = "plots/";
  char suffix[] = ".bmp";

  int M = m*2;
  int row_size, padding, filesize;
  int rowsize_M;
  char *img;

  int i, j, offset;
  int r, g, b;

  char bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
  char bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};

  double constant_inside, denominator;

  strcat(prefix, filename);
  strcat(prefix, suffix);
  fp = fopen(prefix, "wb");
  if (fp == NULL) {
    fprintf(stderr, "Can't open output file %s!\n", prefix);
    exit(1);
  }

  /* The length of each line must be a multiple of 4 bytes */
  row_size = 3 * n;
  // if the file width (3*n) is not a multiple of 4 adds enough bytes to make it
  // a multiple of 4
  padding = (4 - (row_size) % 4) % 4;
  row_size += padding;

  rowsize_M = row_size * M;
  filesize = 54 + rowsize_M;

  img = (char *) malloc(rowsize_M * sizeof(char));
  for (i = 0; i < rowsize_M; ++i) img[i] = 0;

  // Some constants used with smooth_fast_colormap
  constant_inside = 1.75 - log2(0.5);
  denominator = log2((double)maxiter);
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      offset = i * n + j;
      smooth_fast_colormap(offset, maxiter, iters, res2, ims2, &r, &g, &b,
                           constant_inside, denominator);

      offset = row_size * i + 3 * j;
      img[offset++] = b;
      img[offset++] = g;
      img[offset] = r;

      // Symmetric draw
      offset = row_size * (M-1-i) + 3 * j;
      img[offset++] = b;
      img[offset++] = g;
      img[offset] = r;
    }
  }

  bmpfileheader[2] = filesize;
  bmpfileheader[3] = filesize >> 8;
  bmpfileheader[4] = filesize >> 16;
  bmpfileheader[5] = filesize >> 24;

  bmpinfoheader[4]  = n;
  bmpinfoheader[5]  = n >> 8;
  bmpinfoheader[6]  = n >> 16;
  bmpinfoheader[7]  = n >> 24;
  bmpinfoheader[8]  = M;
  bmpinfoheader[9]  = M >> 8;
  bmpinfoheader[10] = M >> 16;
  bmpinfoheader[11] = M >> 24;
  bmpinfoheader[20] = (filesize - 54);
  bmpinfoheader[21] = (filesize - 54) >> 8;
  bmpinfoheader[22] = (filesize - 54) >> 16;
  bmpinfoheader[23] = (filesize - 54) >> 24;

  fwrite(bmpfileheader, 1, 14, fp);
  fwrite(bmpinfoheader, 1, 40, fp);

  fwrite(img, 1, rowsize_M, fp);

  fclose(fp);
  return 0;
}

