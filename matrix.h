/* matrix.h */

/* make sure this header file has not been included already */
#ifndef __MATRIX
#define __MATRIX 1

typedef unsigned char byte;

typedef struct matrix_s {
  int num_rows;
  int num_cols;
  byte* data_ptr;
} matrix_type;

typedef struct matrix_row_s {
  int num_cols;
  byte* data_ptr;
} matrix_row_type;

/* allocate and initialize to 0 a num_rows x num_cols matrix */
void matrix_init (matrix_type* mat_ptr, int num_rows, int num_cols);

/* free up a matrix */
void matrix_deinit (matrix_type* mat_ptr);

/* read a matrix from stdin */
void matrix_read (matrix_type* mat_ptr);

/* read a matrix from a binary file */
void matrix_read_bin (matrix_type* mat_ptr, char* filename, int header_size);

/* print a matrix to stdout */
void matrix_print (matrix_type* mat_ptr);

/* get a row of the matrix (not a copy just a pointer into the data) */
void matrix_get_row (matrix_type* mat_ptr, matrix_row_type* row_ptr,
		     int row);

/* print a row of the matrix */
void matrix_row_print (matrix_row_type* row_ptr);

/* calculate the distance squared between two row vectors */
int matrix_row_dist_sq (matrix_row_type* row1_ptr,
			matrix_row_type* row2_ptr);

#endif
