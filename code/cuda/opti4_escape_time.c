#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef DEBUG
  #include <time.h>
#endif

/******************************************************************************/
static const double X_MAX = 1.0;
static const double X_MIN = -2.5;
static const double Y_MAX = 1.0;
static const double Y_MIN = -1.0;

/******************************************************************************/
extern "C"
void kernel_wrapper(char *h_img, int d_img_size, int MAX_ITER, double X_MIN,
                    double Y_MIN, double h_x_step, double h_y_step,
                    int N, int WIDTH, int row_size);

/******************************************************************************/

int main(int argc, char *argv[]) {

  char *filename;
  FILE *fp;
  char prefix[] = "plots/";
  char suffix[] = ".bmp";

  int M;
  int row_size, padding, filesize;
  int h_img_size, d_img_size;
  int r, g, b;
  int i, j, offset;

  char bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
  char bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};

  int WIDTH = 1920;
  int HEIGHT = 1080;
  int MAX_ITER = 1000;

  char *h_img;
  double h_x_step, h_y_step;

  int N;

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

  M = HEIGHT;

  HEIGHT /= 2; // Symmetry with respect to the x axis

  N = WIDTH * HEIGHT;

  /* The length of each line must be a multiple of 4 bytes */
  row_size = 3 * WIDTH;
  // if the file width (3*n) is not a multiple of 4 adds enough bytes to make it
  // a multiple of 4
  padding = (4 - (row_size) % 4) % 4;
  row_size += padding;

  d_img_size = row_size * HEIGHT;
  h_img_size = d_img_size + d_img_size;


  // Allocate memory on the host and on the device
  h_img = (char *) malloc(h_img_size); // double size on the host

  /****************************************************************************/
  /*                           EVALUATE ESCAPE TIMES                          */
  /****************************************************************************/
  kernel_wrapper(h_img, d_img_size, MAX_ITER, X_MIN, Y_MIN, h_x_step, h_y_step,
                 N, WIDTH, row_size);

  /****************************************************************************/
  /*                            DRAW MANDELBROT SET                           */
  /****************************************************************************/
  for (i = 0; i < HEIGHT; ++i) {
    for (j = 0; j < WIDTH; ++j) {
      offset = row_size * i + 3 * j;
      b = h_img[offset++];
      g = h_img[offset++];
      r = h_img[offset];

      // Symmetric draw
      offset = row_size * (M-1-i) + 3 * j;
      h_img[offset++] = b;
      h_img[offset++] = g;
      h_img[offset] = r;
    }
  }

  strcat(prefix, filename);
  strcat(prefix, suffix);
  fp = fopen(prefix, "wb");
  if (fp == NULL) {
    fprintf(stderr, "Can't open output file %s!\n", prefix);
    exit(1);
  }

  filesize = 54 + h_img_size;

  bmpfileheader[2] = filesize;
  bmpfileheader[3] = filesize >> 8;
  bmpfileheader[4] = filesize >> 16;
  bmpfileheader[5] = filesize >> 24;

  bmpinfoheader[4]  = WIDTH;
  bmpinfoheader[5]  = WIDTH >> 8;
  bmpinfoheader[6]  = WIDTH >> 16;
  bmpinfoheader[7]  = WIDTH >> 24;
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

  fwrite(h_img, 1, h_img_size, fp);

  fclose(fp);

  /****************************************************************************/
  /*                              FREE RESOURCES                              */
  /****************************************************************************/
  free(h_img);


  #ifdef DEBUG
    fprintf(stdout, "Execution time = %f [s]\n",
            (double)(clock() - begin) / CLOCKS_PER_SEC);
  #endif

  return 0;
}