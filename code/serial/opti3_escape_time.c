#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "io_binary.h"
#ifdef DEBUG
  #include <time.h>
#endif

/******************************************************************************/
#define X_MAX 1.0
#define X_MIN -2.5
#define Y_MAX 1.0
#define Y_MIN -1.0

/******************************************************************************/
void allocate_int_grid(int **vec, int size);
void allocate_double_grid(double **vec, int size);
void deallocate_int_grid(int **vec);
void deallocate_double_grid(double **vec);
void in_cardioid_or_period2_bulb(int *iterp, double x, double y, int maxiter);
void compute_escape_time(int **iters, double **res2, double **ims2,
                         int i, int j, int width,
                         double x_step, double y_step, int maxiter);

/******************************************************************************/

int main(int argc, char *argv[]) {
  double x_step, y_step;
  int i, j;
  int *iters;
  double *res2, *ims2;

  char *filename;
  int WIDTH = 1920;
  int HEIGHT = 1080;
  int MAX_ITER = 1000;

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

  x_step = (X_MAX - X_MIN) / WIDTH;
  y_step = (Y_MAX - Y_MIN) / HEIGHT;

  HEIGHT /= 2; // Symmetry with respect to the x axis

  allocate_int_grid(&iters, WIDTH * HEIGHT);
  allocate_double_grid(&res2, WIDTH * HEIGHT);
  allocate_double_grid(&ims2, WIDTH * HEIGHT);

  /****************************************************************************/
  /*                           EVALUATE ESCAPE TIMES                          */
  /****************************************************************************/
  for(i = 0; i < HEIGHT; ++i) {
    for(j = 0; j < WIDTH; ++j) {
      compute_escape_time(&iters,&res2,&ims2,i,j,WIDTH,x_step,y_step,MAX_ITER);
    }
  }

  /****************************************************************************/
  /*                            DRAW MANDELBROT SET                           */
  /****************************************************************************/
  draw_fast_mandelbrot(HEIGHT,WIDTH,&iters,&res2,&ims2,MAX_ITER,filename);

  /****************************************************************************/
  /*                              FREE RESOURCES                              */
  /****************************************************************************/
  deallocate_int_grid(&iters);
  deallocate_double_grid(&res2);
  deallocate_double_grid(&ims2);

  #ifdef DEBUG
    fprintf(stdout, "Execution time = %f [s]\n",
            (double)(clock() - begin) / CLOCKS_PER_SEC);
  #endif

  return 0;
}


/******************************************************************************/
void allocate_int_grid(int **vec, int size) {
  *vec = (int *) malloc(size*sizeof(int));
}

void allocate_double_grid(double **vec, int size) {
  *vec = (double *) malloc(size*sizeof(double));
}

void deallocate_int_grid(int **vec) {
  free(*vec);
}

void deallocate_double_grid(double **vec) {
  free(*vec);
}


void in_cardioid_or_period2_bulb(int *iterp, double x, double y, int maxiter)
{
  double xdiff = x - 0.25;
  double y2 = y * y;
  double q = xdiff*xdiff + y2;

  // Is the point in the cardioid?
  if (q * (q + xdiff) < 0.25*y2) {
    *iterp = maxiter;
  }
  else if ((x+1.)*(x+1.) + y2 < 0.0625) { // Is the point in the period-2 bulb?
    *iterp = maxiter;
  }

}


void compute_escape_time(int **iters, double **res2, double **ims2,
                         int i, int j, int width,
                         double x_step, double y_step, int maxiter)
{
  int offset = i * width + j;
  int iteration = 0;
  double c_re = X_MIN + x_step/2 + j*x_step;
  double c_im = Y_MIN + y_step/2 + i*y_step;
  double zn_re = 0.;
  double zn_im = 0.;
  double tmp_re;
  double re2 = 0.;
  double im2 = 0.;
  int bailout_radius2 = 2*2;

  // Check if point is in cardioid or in period-2 bulb
  in_cardioid_or_period2_bulb(&iteration, c_re, c_im, maxiter);

  while ((re2 + im2 < bailout_radius2) && (iteration < maxiter)) {
    tmp_re = re2 - im2 + c_re;
    zn_im = zn_re * zn_im;
    zn_im += zn_im; // Multiply by two
    zn_im += c_im;
    zn_re = tmp_re;

    re2 = zn_re * zn_re;
    im2 = zn_im * zn_im;
    iteration++;
  }

  (*iters)[offset] = iteration;
  (*res2)[offset] = re2;
  (*ims2)[offset] = im2;
}
