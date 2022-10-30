/* Demo for linear optimization algorithms with/without preconditioning
 * - conjugate gradient method
 * - BiCGStab (Bi-conjugate gradient with stabalization)
 * - preconditioned BiCGStab
 * - GMRES (generalized minimum residual)
 * - GMRES with left and right preconditioning
 *------------------------------------------------------------------------
 *
 * Copyright (c) 2020-2022 Harbin Institute of Technology. All rights reserved.
 * Author: Pengliang Yang 
 * Email: ypl.2100@gmail.com
 * Homepage: https://yangpl.wordpress.com
 *-----------------------------------------------------------------------*/
#ifndef __sparse_h__
#define __sparse_h__

typedef struct{
  int nrow;
  int ncol;
  int nnz;//number of non-zeros
  int *row;//row indices
  int *col;//column indices
  double *val;//values of the matrix A
} coo_t;//COOrdinate format


typedef struct{
  int nrow;
  int ncol;
  int nnz;
  double *val;//values in the matrix
  int *row_ptr;//row pointer, row_ptr[i]=row_ptr[i-1] +nnz in row-i; row_ptr[0]=0
  int *col_ind;//column indices
}csr_t;//compressed sparse column/compressed row storage (CSR/CRS) format 


typedef struct{
  int nrow;
  int ncol;
  int nnz;
  double *val;//values in the matrix
  int *row_ind;//row indices
  int *col_ptr;//column pointer
}csc_t;//compressed sparse column

#endif
