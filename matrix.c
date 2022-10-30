/* build matrix associated with 2D Laplace operator
 *------------------------------------------------------------------------
 *
 * Copyright (c) 2020-2022 Harbin Institute of Technology. All rights reserved.
 * Author: Pengliang Yang 
 * Email: ypl.2100@gmail.com
 * Homepage: https://yangpl.wordpress.com
 *-----------------------------------------------------------------------*/
#include "cstd.h"
#include "sparse.h"

/*==========================================================*/
double **A;//standard 2D matrix

void A_init(int n)
{
  int i, j;
  
  A = alloc2double(n, n);
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++){ 
      A[i][j] = fabs(i-j);
    }
  }
}

void A_close()
{
  free2double(A);
}
//standard 2D matrix applied to vector x: Ax=A*x
void A_apply(int n, double *x, double *Ax)
{
  int i, j;
  
  for(i=0; i<n; i++){
    Ax[i] = 0;
    for(j=0; j<n; j++) Ax[i] += A[i][j]*x[j];
  }
}

//standard 2D matrix applied to vector y: Aty=At*y
void At_apply(int n, double *y, double *Aty)
{
  int i, j;

  memset(Aty, 0, n*sizeof(double));
  for(i=0; i<n; i++){
    for(j=0; j<n; j++) Aty[j] += A[i][j]*y[i];
  }
}


/*===========================================================*/
coo_t Acoo;//matrix in coo format

//build matrix in coo format for laplace operator A=-Dx^2 - Dy^2
void Acoo_init(int nx, int ny, double dx, double dy)
{
  int i, j, k;
  double _dx2, _dy2;

  //count the number of nonzeros in sparse banded matrix A
  k = 0;
  for(j=0; j<=ny; j++){
    for(i=0; i<=nx; i++){
      //the diagonal element A(i,j)
      k++;
      
      //the off-diagonal element
      if(j-1>=0){//element A(i,j-1)
	k++;	
      }
      if(i-1>=0){//element A(i-1,j)
	k++;	
      }
      if(i+1<=nx){//element A(i+1,j)
	k++;
      }
      if(j+1<=ny){//element A(i,j+1)
	k++;
      }      
    }
  }
  Acoo.nnz = k;//number of non-zeros
  Acoo.nrow = (nx+1)*(ny+1);
  Acoo.ncol = (nx+1)*(ny+1);
  Acoo.row = alloc1int(Acoo.nnz);
  Acoo.col = alloc1int(Acoo.nnz);
  Acoo.val = alloc1double(Acoo.nnz);

  _dx2 = 1./(dx*dx);
  _dy2 = 1./(dy*dy);
  k = 0;
  for(j=0; j<=ny; j++){
    for(i=0; i<=nx; i++){
      //the diagonal element A(i,j)
      Acoo.val[k] = -2.*(_dx2 + _dy2);
      Acoo.row[k] = i + (nx+1)*j;
      Acoo.col[k] = i + (nx+1)*j;
      k++;
      
      //the off-diagonal element
      if(i-1>=0){//element A(i-1,j)
	Acoo.val[k] = _dx2;
	Acoo.row[k] = i + (nx+1)*j;
	Acoo.col[k] = (i-1) + (nx+1)*j;
	k++;	
      }
      if(i+1<=nx){//element A(i+1,j)
	Acoo.val[k] = _dx2;
	Acoo.row[k] = i + (nx+1)*j;
	Acoo.col[k] = (i+1) + (nx+1)*j;
	k++;
      }
      if(j-1>=0){//element A(i,j-1)
	Acoo.val[k] = _dy2;
	Acoo.row[k] = i + (nx+1)*j;
	Acoo.col[k] = i + (nx+1)*(j-1);
	k++;	
      }
      if(j+1<=ny){//element A(i,j+1)
	Acoo.val[k] = _dy2;
	Acoo.row[k] = i + (nx+1)*j;
	Acoo.col[k] = i + (nx+1)*(j+1);
	k++;
      }      
    }
  }
}

void Acoo_close()
{
  free1int(Acoo.row);
  free1int(Acoo.col);
  free1double(Acoo.val);
}

//compute y=Ax by applying operator A in coo format
void Acoo_apply(int n, double *x, double *y)
{
  int k;

  memset(y, 0, n*sizeof(double));  
  /* y_i += a_{ij}*x_i where i=Acoo.row[k], j=Acoo.col[k], a_{ij}=Acoo.val[k] */
  for(k=0; k<Acoo.nnz; k++) y[Acoo.row[k]] += Acoo.val[k]*x[Acoo.col[k]];
}

/*=================================================================*/
csr_t Acsr;//matrix in CSR format

//build matrix in coo format for laplace operator A=-Dx^2 - Dy^2
void Acsr_init(int nx, int ny, double dx, double dy)
{
  int i, j, k, row_ind;
  double _dx2, _dy2;

  //count the number of nonzeros in sparse banded matrix A
  k = 0;
  for(j=0; j<=ny; j++){
    for(i=0; i<=nx; i++){
      //the diagonal element A(i,j)
      k++;
      
      //the off-diagonal element
      if(j-1>=0){//element A(i,j-1)
	k++;	
      }
      if(i-1>=0){//element A(i-1,j)
	k++;	
      }
      if(i+1<=nx){//element A(i+1,j)
	k++;
      }
      if(j+1<=ny){//element A(i,j+1)
	k++;
      }      
    }
  }
  Acsr.nnz = k;//number of non-zeros
  Acsr.nrow = (nx+1)*(ny+1);
  Acsr.ncol = (nx+1)*(ny+1);
  Acsr.row_ptr = alloc1int(Acsr.nrow+1);//count number of nonzeros in each row first
  Acsr.col_ind = alloc1int(Acsr.nnz);
  Acsr.val = alloc1double(Acsr.nnz);

  memset(Acsr.row_ptr, 0, (Acsr.nrow+1)*sizeof(int));
  _dx2 = 1./(dx*dx);
  _dy2 = 1./(dy*dy);
  k = 0;
  for(j=0; j<=ny; j++){
    for(i=0; i<=nx; i++){
      //the diagonal element A(i,j)
      row_ind =  i + (nx+1)*j;
      Acsr.row_ptr[row_ind+1]++;//increase counter
      Acsr.col_ind[k] = i + (nx+1)*j;
      Acsr.val[k] = -2.*(_dx2 + _dy2);
      k++;
      
      //the off-diagonal element
      if(i-1>=0){//element A(i-1,j)
	row_ind = i + (nx+1)*j;
	Acsr.row_ptr[row_ind+1]++;//increase counter
	Acsr.col_ind[k] = (i-1) + (nx+1)*j;
	Acsr.val[k] = _dx2;
	k++;	
      }
      if(i+1<=nx){//element A(i+1,j)
	row_ind = i + (nx+1)*j;
	Acsr.row_ptr[row_ind+1]++;//increase counter
	Acsr.col_ind[k] = (i+1) + (nx+1)*j;
	Acsr.val[k] = _dx2;
	k++;
      }
      if(j-1>=0){//element A(i,j-1)
	row_ind = i + (nx+1)*j;
	Acsr.row_ptr[row_ind+1]++;//increase counter
	Acsr.col_ind[k] = i + (nx+1)*(j-1);
	Acsr.val[k] = _dy2;
	k++;	
      }
      if(j+1<=ny){//element A(i,j+1)
	row_ind = i + (nx+1)*j;
	Acsr.row_ptr[row_ind+1]++;//increase counter
	Acsr.col_ind[k] = i + (nx+1)*(j+1);
	Acsr.val[k] = _dy2;
	k++;
      }      
    }
  }

  /* in CSR format: row_ptr[i] stores nnz upto but not including i-th row */
  /* then add total number of nonzeros before i-th row */
  for(i=0; i<Acsr.nrow; i++) Acsr.row_ptr[i+1] += Acsr.row_ptr[i];
}

void Acsr_close()
{
  free1int(Acsr.row_ptr);
  free1int(Acsr.col_ind);
  free1double(Acsr.val);
}

//y=Ax
void Acsr_apply(int n, double *x, double *y)
{
  int i, j, k;

  for(i=0; i<n; i++){
    y[i] = 0;
    /* y_i += a_ij*x_j where j=Acsr.col_ind[k], a_ij=Acsr.val[k] */
    for(k=Acsr.row_ptr[i]; k<Acsr.row_ptr[i+1]; k++){
      j = Acsr.col_ind[k];//a_ij=Acsr.val[k]
      y[i] += Acsr.val[k]*x[j];//y_i += a_ij*x_j
    }
  }
}

//y=A^T x
void Acsr_transpose_apply(int n, double *x, double *y)
{
  int i, j, k;

  for(i=0; i<n; i++){
    y[i] = 0;
    for(k=Acsr.row_ptr[i]; k<Acsr.row_ptr[i+1]; k++){
      j = Acsr.col_ind[k];//a_ij=Acsr.val[k] 
      y[j] += Acsr.val[k]*x[i];
    }
  }
}



/*==============================================================*/
csc_t Acsc;//matrix in CSC format

void Acsc_close()
{
  free1int(Acsc.row_ind);
  free1int(Acsc.col_ptr);
  free1double(Acsc.val);
}

//y=Ax
void Acsc_apply(int n, double *x, double *y)
{
  int i, j, k;

  memset(y, 0, n*sizeof(double));
  for(j=0; j<n; j++){
    /* y_i += a_ij*x_j where j=Acsc.col_ind[k], a_ij=Acsc.val[k], i.e.
     * the column index of the k-th element in matrix Acsc.val[] is j */
    for(k=Acsc.col_ptr[j]; k<Acsc.col_ptr[j+1]; k++){
      i = Acsc.row_ind[k];
      y[i] += Acsc.val[k]*x[j];//y_i += a_ij*x_j
    }
  }
}

//==============================================================
//convert coo to csr format
void coo2csr()
{
  int i, k;

  Acsr.nrow = Acoo.nrow;
  Acsr.ncol = Acoo.ncol;
  Acsr.nnz = Acoo.nnz;
  Acsr.val = alloc1double(Acsr.nnz);
  Acsr.col_ind = alloc1int(Acsr.nnz);
  Acsr.row_ptr = alloc1int(Acoo.nrow+1);

  /* in coo format: a_ij=Acoo.val[k], where i=Acoo.row[k], j=Acoo.col[k] */
  memset(Acsr.row_ptr, 0, (Acoo.nrow+1)*sizeof(int));
  for(k=0; k<Acsr.nnz; k++){
    Acsr.val[k] = Acoo.val[k];//a_ij=Acoo.val[k]
    Acsr.col_ind[k] = Acoo.col[k];//j=Acoo.col[k]
    Acsr.row_ptr[Acoo.row[k]+1]++;//count number of nonzeros in each row first
  }
  /* in CSR format: row_ptr[i] stores nnz upto but not including i-th row */
  /* then add total number of nonzeros before i-th row */
  for(i=0; i<Acoo.nrow; i++) Acsr.row_ptr[i+1] += Acsr.row_ptr[i];
}

//convert csr to coo format
void csr2coo()
{
  int i, j, k;

  Acoo.nrow = Acsr.nrow;
  Acoo.ncol = Acsr.ncol;
  Acoo.nnz = Acsr.nnz;
  Acoo.row = alloc1int(Acoo.nnz);
  Acoo.col = alloc1int(Acoo.nnz);
  Acoo.val = alloc1double(Acoo.nnz);
  
  for(i=0; i<Acsr.nrow; i++){
    for(k=Acsr.row_ptr[i]; k<Acsr.row_ptr[i]; k++){
      j = Acsr.col_ind[k];
      Acoo.row[k] = i;
      Acoo.col[k] = j;
      Acoo.val[k] = Acsr.val[k];      
    }
  }

}


//convert csr to csc format
void csr2csc()
{
  int i, j, k, m;
  int *pos;
  
  Acsc.nrow = Acsr.nrow;
  Acsc.ncol = Acsr.ncol;
  Acsc.col_ptr = alloc1int(Acsc.ncol+1);

  memset(Acsc.col_ptr, 0, (Acsc.ncol+1)*sizeof(int));
  for(i=0; i<Acsc.nrow; i++){
    for(k=Acsr.row_ptr[i]; k<Acsr.row_ptr[i+1]; k++){
      j = Acsr.col_ind[k];//a_ij=Acsr.val[k]
      Acsc.col_ptr[j+1] += 1;//first, count the number of nonzeros in each column of A
    }
  }
  //then, add the total number of nonzeros before j-th column
  for(j=0; j<Acsc.ncol; j++) Acsc.col_ptr[j+1] += Acsc.col_ptr[j];
  
  Acsc.nnz = Acsr.row_ptr[Acsr.nrow];
  Acsc.val = alloc1double(Acsc.nnz);
  Acsc.row_ind = alloc1int(Acsc.nnz);
  
  pos = alloc1int(Acsc.ncol);//record current available position in each column
  memset(pos, 0, Acsc.ncol*sizeof(int));
  for(i=0; i<Acsr.nrow; i++){
    for(k=Acsr.row_ptr[i]; k<Acsr.row_ptr[i+1]; k++){
      j = Acsr.col_ind[k];//a_ij=Acsr.val[k]
      //add number of nonzeros before j-th column and number of nonzeros in j-th column
      m = Acsc.col_ptr[j] + pos[j];
      Acsc.val[m] = Acsr.val[k];//so the m-th element in Acsc.val[] is: a_ij=Acsr.val[k]
      Acsc.row_ind[m] = i;
      pos[j]++;//increase the counter
    }
  }

  free1int(pos);
}

void gauss_seidel_iterations(int n, double *x, double *b, int niter, double tol)
{
  int i, j, iter;
  double s, tmp;
  double *Ax;
  Ax = alloc1double(n);
  
  for(iter=0; iter<niter; iter++){
    A_apply(n, x, Ax);
    s = 0;
    for(i=0; i<n; i++){
      tmp = b[i]-Ax[i];
      s += tmp*tmp;
    }
    s = sqrt(s);
    if(s<tol) break;
    printf("iter=%d err=%e\n", iter, s);

    for(i=0; i<n; i++){
      x[i] = b[i];
      for(j=0; j<n; j++)
	if(j!=i) x[i] -= A[i][j]*x[j];
      x[i] /= A[i][i];//diagonals are zeros, making algorithm generating nan
    }
  }

  free1double(Ax);
}
