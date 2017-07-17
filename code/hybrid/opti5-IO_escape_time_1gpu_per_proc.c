#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/******************************************************************************/
static const float X_MAX = 1.0;
static const float X_MIN = -2.5;
static const float Y_MAX = 1.0;
static const float Y_MIN = -1.0;

/******************************************************************************/
extern
void kernel_wrapper(char *h_img, int d_img_size, int MAX_ITER, float X_MIN,
                    float Y_MIN, float h_x_step, float h_y_step,
                    int N, int N_local, int mpi_row_offset, int prank,
                    int WIDTH, int row_size);

void swap_rows(char **h_vec, int m, int n);

/******************************************************************************/

int main(int argc, char *argv[]) {

  char *filename;
  char prefix[] = "plots/";
  char suffix[] = ".bmp";

  int M;
  int row_size, padding, filesize;
  int h_img_size, d_img_size;
  // int r, g, b;
  // int i, j;
  int offset;

  char bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
  char bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};

  int prank, psize;
  int m_local, N_local, mpi_row_offset; // offset in the row number
  MPI_Status status;
  MPI_File fh;

  int WIDTH = 1920;
  int HEIGHT = 1080;
  int MAX_ITER = 1000;

  char *h_img;
  float h_x_step, h_y_step;

  int N;


  /****************************************************************************/
  /*                              INITIIALIZE MPI                             */
  /****************************************************************************/
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

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
    double start = MPI_Wtime();
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

  // Divide HEIGHT in psize and distribute the excess to the (n % psize) proc
  m_local = HEIGHT / psize + (prank < HEIGHT % psize ? 1 : 0);
  // Computing the offset of where in the global array the local array is
  // located.
  mpi_row_offset = (HEIGHT/psize)*prank+(prank<HEIGHT%psize?prank:HEIGHT%psize);

  d_img_size = row_size * m_local;
  h_img_size = d_img_size;

  N_local = WIDTH * m_local;

  // Allocate memory on the host
  h_img = (char *) malloc(h_img_size); // double size on the host

  /****************************************************************************/
  /*                           EVALUATE ESCAPE TIMES                          */
  /****************************************************************************/
  kernel_wrapper(h_img, d_img_size, MAX_ITER, X_MIN, Y_MIN, h_x_step, h_y_step,
                 N, N_local, mpi_row_offset, prank, WIDTH, row_size);

  /****************************************************************************/
  /*                            DRAW MANDELBROT SET                           */
  /****************************************************************************/
  strcat(prefix, filename);
  strcat(prefix, suffix);

  filesize = 54 + row_size*M;

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

  // opening a file in write and create mode
  MPI_File_open(MPI_COMM_WORLD, prefix,
                MPI_MODE_WRONLY | MPI_MODE_CREATE,
                MPI_INFO_NULL, &fh);
  // defining the size of the file
  MPI_File_set_size(fh, filesize);

  // rank 0 writes the header
  if (prank == 0) {
    MPI_File_write_at(fh, 0, bmpfileheader, 14, MPI_CHAR, &status);
    MPI_File_write_at(fh, 14, bmpinfoheader, 40, MPI_CHAR, &status);
  }

  offset = 54 + row_size * mpi_row_offset;

  // We also could write that data with a write_at, the view is just to show
  // different possibilities
  MPI_File_set_view(fh, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
  MPI_File_write(fh, h_img, h_img_size, MPI_CHAR, &status);

  swap_rows(&h_img, m_local, row_size);
  offset = 54 + row_size * (M - mpi_row_offset - m_local);
  MPI_File_set_view(fh, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
  MPI_File_write(fh, h_img, h_img_size, MPI_CHAR, &status);

  MPI_File_close(&fh);

  #ifdef DEBUG
    double ttime = MPI_Wtime() - start;
    fprintf(stdout, "Execution time = %f [s]\n", ttime);
  #endif


  /****************************************************************************/
  /*                              FREE RESOURCES                              */
  /****************************************************************************/
  free(h_img);
  MPI_Finalize();

  return 0;
}


/******************************************************************************/
void swap_rows(char **h_vec, int m, int n) {
  int i, j;
  int half_m = m / 2;
  int offset1, offset2;
  char tmp;

  for (i = 0; i < half_m; ++i) {
    offset1 = i * n;
    offset2 = (m - 1 - i) * n;
    for (j = 0; j < n; ++j) {
      tmp = (*h_vec)[offset1];
      (*h_vec)[offset1++] = (*h_vec)[offset2];
      (*h_vec)[offset2++] = tmp;
    }
  }
}
