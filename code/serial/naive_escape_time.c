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
void deallocate_int_grid(int **vec);
void compute_escape_time(int **iters, int i, int j, int width,
                         double x_step, double y_step, int maxiter);

/******************************************************************************/

int main(int argc, char *argv[]) {
  double x_step, y_step;
  int i, j;
  int *iters;

  char *filename;
  int WIDTH = 1920;
  int HEIGHT = 1080;
  int MAX_ITER = 1000;

  /****************************************************************************/
  /*                            GET USER PARAMETERS                           */
  /****************************************************************************/
  if (argc == 2) {
    filename = argv[1];
  }
  else if (argc == 4) {
    filename = argv[1];
    WIDTH = atoi(argv[2]);
    HEIGHT = atoi(argv[3]);
  }
  else if (argc == 5) {
    filename = argv[1];
    WIDTH = atoi(argv[2]);
    HEIGHT = atoi(argv[3]);
    MAX_ITER = atoi(argv[4]);
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

  allocate_int_grid(&iters, WIDTH * HEIGHT);

  /****************************************************************************/
  /*                           EVALUATE ESCAPE TIMES                          */
  /****************************************************************************/
  x_step = (X_MAX - X_MIN) / WIDTH;
  y_step = (Y_MAX - Y_MIN) / HEIGHT;

  for(i = 0; i < HEIGHT; ++i) {
    for(j = 0; j < WIDTH; ++j) {
      compute_escape_time(&iters, i, j, WIDTH, x_step, y_step, MAX_ITER);
    }
  }

  /****************************************************************************/
  /*                            DRAW MANDELBROT SET                           */
  /****************************************************************************/
  draw_mandelbrot(HEIGHT, WIDTH, &iters, MAX_ITER, filename);

  /****************************************************************************/
  /*                              FREE RESOURCES                              */
  /****************************************************************************/
  deallocate_int_grid(&iters);

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

void deallocate_int_grid(int **vec) {
  free(*vec);
}


void compute_escape_time(int **iters, int i, int j, int width,
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

  while ((re2 + im2 < 2*2) && (iteration < maxiter)) {
    tmp_re = re2 - im2 + c_re;
    zn_im = 2*zn_re*zn_im + c_im;
    zn_re = tmp_re;

    re2 = zn_re*zn_re;
    im2 = zn_im*zn_im;
    iteration++;
  }

  (*iters)[offset] = iteration;
}
