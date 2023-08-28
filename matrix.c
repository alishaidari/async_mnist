#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

/* allocate and initialize to 0 a num_rows x num_cols matrix */
void matrix_init (matrix_type* mat_ptr, int num_rows, int num_cols) {
  mat_ptr->num_rows = num_rows;
  mat_ptr->num_cols = num_cols;
  mat_ptr->data_ptr = (byte*) calloc (num_rows*num_cols, sizeof(byte));
  #ifdef DEBUG
  if (mat_ptr->data_ptr == 0) {
    printf ("error in matrix_init : calloc returned 0.\n");
    exit(1);
  }
  #endif
}

/* free up a matrix */
void matrix_deinit (matrix_type* mat_ptr) {
  mat_ptr->num_rows = 0;
  mat_ptr->num_cols = 0;
  free(mat_ptr->data_ptr);
  mat_ptr->data_ptr = 0;
}

/* read a matrix from stdin */
void matrix_read (matrix_type* mat_ptr) {
  int count = 0;
  byte next;
  while (scanf("%hhu",&next) == 1) {
    if (count >= mat_ptr->num_rows*mat_ptr->num_cols) {
      printf ("ERROR: stdin contains too many numbers\n");
      exit(1);
    }
    mat_ptr->data_ptr[count] = next;
    count += 1;
  }
  if (count < mat_ptr->num_rows*mat_ptr->num_cols) {
    printf ("ERROR: stdin contains too few numbers\n");
    exit(1);    
  }
}

/* read a matrix from a binary file */
void matrix_read_bin (matrix_type* mat_ptr, char* filename, int header_size) {
    byte header[header_size];
    FILE* fptr;
    int data_size = mat_ptr->num_rows*mat_ptr->num_cols;
    size_t bytes_read = 0;
    /* open the binary file for reading */
    fptr = fopen(filename,"rb"); 
    /* need to check for null */
    if (fptr == 0) {
	printf ("Error opening binary data file.\n");
	exit(1);
    }
    /* skip over header */
    bytes_read = fread(header, sizeof(byte), header_size, fptr);
    /* read data */
    bytes_read = fread(mat_ptr->data_ptr, sizeof(byte), data_size, fptr);
    /* close the binary file */
    fclose(fptr);
}

/* print a matrix to stdout */
void matrix_print (matrix_type* mat_ptr) {
  int i;
  matrix_row_type row;
  printf ("number of rows = %d\n",mat_ptr->num_rows);
  printf ("number of columns = %d\n",mat_ptr->num_cols);
  for (i=0;i<mat_ptr->num_rows;i++) {
    matrix_get_row (mat_ptr,&row,i);
    matrix_row_print (&row);
  }
}

/* get a row of the matrix (not a copy just a pointer into the data) */
void matrix_get_row (matrix_type* mat_ptr, matrix_row_type* row_ptr,
		     int row) {
  #ifdef DEBUG
  if ((row < 0) || (row >= mat_ptr->num_rows)) {
    printf ("error in matrix_get_row : row out of range\n");
    exit(1);
  }
  #endif
  row_ptr->num_cols = mat_ptr->num_cols;
  row_ptr->data_ptr = &(mat_ptr->data_ptr[row*mat_ptr->num_cols]);
}

/* print a row of the matrix */
void matrix_row_print (matrix_row_type* row_ptr) {
  int i;
  for (i=0;i<row_ptr->num_cols;i++) {
    printf ("%hhu ",row_ptr->data_ptr[i]);
  }
  printf ("\n");
}

/* calculate the distance squared between two row vectors */
int matrix_row_dist_sq (matrix_row_type* row1_ptr,
			matrix_row_type* row2_ptr) {
  #ifdef DEBUG
  if (row1_ptr->num_cols != row2_ptr->num_cols) {
    printf ("error in matrix_row_dist_sq: number of columns mismatch\n");
  }
  #endif
  int i;
  int dist_sq = 0;
  int comp_diff = 0; /* component difference */
  for (i=0;i<row1_ptr->num_cols;i++) {
    comp_diff = row1_ptr->data_ptr[i]-row2_ptr->data_ptr[i];
    dist_sq += comp_diff*comp_diff;
  }
  return dist_sq;
}

