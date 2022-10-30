/* Demo for linear optimization algorithms with/without preconditioning
 * - conjugate gradient method
 * - BiCGStab (Bi-conjugate gradient with stabalization)
 * - preconditioned BiCGStab
 * - GMRES (generalized minimum residual)
 * - GMRES with left and right preconditioning
 * - CGNR
 *------------------------------------------------------------------------
 *
 * Copyright (c) 2020-2022 Harbin Institute of Technology. All rights reserved.
 * Author: Pengliang Yang 
 * Email: ypl.2100@gmail.com
 * Homepage: https://yangpl.wordpress.com
 *-----------------------------------------------------------------------*/
#include "cstd.h"
#include "solver.h"

#define PI 3.141592653589793238


//linear solver using conjugate gradient method
void solve_cg(int n, double *x, double *b, op_t Aop, int niter, double tol);
//linear solver using preconditioned conjugate gradient method
void solve_pcg(int n, double *x, double *b, op_t Aop, op_t invMop, int niter, double tol);
//linear solver using BiCGStab
void solve_bicgstab(int n, double *x, double *b, op_t Aop, int niter, double tol);
//linear solver using preconditioned BiCGStab
void solve_pbicgstab(int n, double *x, double *b, op_t Aop, op_t invPop, int niter, double tol);
//preconditioned BiCGStab2
void solve_pbicgstab2(int n, double *x, double *b, op_t Aop, op_t invMop, int niter, double tol);
//GMRES without preconditioning
void solve_gmres(int n, double *x, double *b, op_t Aop, int niter, double tol, int m);
//GMRES with left preconditioning
void solve_gmres_leftpreco(int n, double *x, double *b, op_t Aop, op_t invMop, int niter, double tol, int m);
//GMRES with right preconditioning
void solve_gmres_rightpreco(int n, double *x, double *b, op_t Aop, op_t invMop, int niter, double tol, int m);
void solve_cgnr(int n, double *x, double *b, op_t Aop, op_t Atop, int niter, double tol);

void A_init(int n);
void A_apply(int n, double *x, double *Ax);
void At_apply(int n, double *y, double *Aty);
void A_close();
void gauss_seidel_iterations(int n, double *x, double *b, int niter, double tol);

void Acoo_init(int nx, int ny, double dx, double dy);
void Acoo_apply(int n, double *x, double *y);
void Acoo_close();

void Acsr_init(int nx, int ny, double dx, double dy);
void Acsr_apply(int n, double *x, double *y);
void Acsr_close();

void Acsc_init(int nx, int ny, double dx, double dy);
void Acsc_apply(int n, double *x, double *y);
void Acsc_close();

void coo2csr();
void csr2csc();

//solve Py=p, solution: y=invP*p
void invP_apply(int n, double *p, double *y)
{
  memcpy(y, p, n*sizeof(double));//choose K=I, so y=p
}

//problem 1: 1D signal reconstruction example
int main(int argc, char **argv)
{
  int n, niter, method, m;
  int i, j;
  double *x, *b, *xt, omegas[3];
  double tol, dt;
  FILE *fp;
  
  initargs(argc,argv);
  
  if(!getparint("n", &n)) n = 200;/* lenght of the singal */
  if(!getparint("niter", &niter)) niter = 100;/* maximum number of iterations */
  if(!getpardouble("tol", &tol)) tol = 1e-7;/* maximum number of iterations */
  if(!getparint("method", &method)) method = 1;/*1=CG; 2=BiCGStab; 3=preconditioned BiCGStab */
  if(!getparint("m", &m)) m = 10;/* memory length for GMRES algorithm */
  
  n = 200;
  x = alloc1double(n);
  xt = alloc1double(n);
  b = alloc1double(n);


  dt = 1./n;
  omegas[0] = 2.*PI*0.5;
  omegas[1] = 2.*PI*2.1;
  omegas[2] = 2.*PI*4.9;
  for(j=0; j<n; j++)
    xt[j] = sin(omegas[0]*j*dt) + 0.7*(sin(omegas[1]*j*dt) + sin(omegas[2]*j*dt));//x_true

  A_init(n);
  A_apply(n, xt, b);//b=A*xt
  memset(x, 0, n*sizeof(double));//x=0 as initialization

  /* if(method==0) */
  /*   gauss_seidel_iterations(n, x, b, niter, tol); */
  if(method==1)
    solve_cg(n, x, b, A_apply, niter, tol);
  else if(method==2)
    solve_bicgstab(n, x, b, A_apply, niter, tol);
  else if(method==3)
    solve_pbicgstab(n, x, b, A_apply, invP_apply, niter, tol);
  else if(method==4)
    solve_gmres(n, x, b, A_apply, niter, tol, m);
  else if(method==5)
    solve_gmres_leftpreco(n, x, b, A_apply, invP_apply, niter, tol, m);
  else if(method==6)
    solve_gmres_rightpreco(n, x, b, A_apply, invP_apply, niter, tol, m);
  else if(method==7)
    solve_cgnr(n, x, b, A_apply, At_apply, niter, tol);

  
  /* output true signal and reconstructed one */
  fp= fopen("result.txt", "w");
  fprintf(fp, "x_true \t x_rec \n");
  for(i=0; i<n; i++) fprintf(fp, "%e \t %e\n", xt[i], x[i]);
  fclose(fp);

  free1double(x);
  free1double(b);
  free1double(xt);

  return 0;
}


//problem 2: possion equation \Delta u = f.
int main2(int argc, char **argv)
{
  int nx, ny, n, m, niter, method, i, j, k;
  double dx, dy, tol;
  double *x, *b, *xt;
  FILE *fp;
    
  initargs(argc,argv);
  
  if(!getparint("nx", &nx)) nx = 128;/* lenght of the singal */
  if(!getparint("ny", &ny)) ny = 128;/* lenght of the singal */
  if(!getparint("niter", &niter)) niter = 100;/* maximum number of iterations */
  if(!getpardouble("tol", &tol)) tol = 1e-6;/* maximum number of iterations */
  if(!getparint("method", &method)) method = 1;/*1=CG; 2=BiCGStab; 3=preconditioned BiCGStab */
  if(!getparint("m", &m)) m = 10;/* memory length for GMRES algorithm */

  dx = 1./nx;
  dy = 1./ny;
  n = (nx+1)*(ny+1);

  x = alloc1double(n);
  xt = alloc1double(n);
  b = alloc1double(n);
  
  for(j=0; j<=ny; j++){
    for(i=0; i<=nx; i++){
      k = i + (nx+1)*j;
      xt[k] = 1.+ i-j;//Toeplitz matrix
    }
  }
  memset(x, 0, n*sizeof(double));//x=0 as initialization

    
  // ------ use matrix A stored in coo format --------------
  Acoo_init(nx, ny, dx, dy);
  Acoo_apply(n, xt, b);
  coo2csr();

  if(method==1)
    solve_cg(n, x, b, Acsr_apply, niter, tol);
  else if(method==2)
    solve_bicgstab(n, x, b, Acsr_apply, niter, tol);
  else if(method==3)
    solve_pbicgstab(n, x, b, Acsr_apply, invP_apply, niter, tol);
  else if(method==4)
    solve_gmres(n, x, b, Acsr_apply, niter, tol, m);
  else if(method==5)
    solve_gmres_leftpreco(n, x, b, Acsr_apply, invP_apply, niter, tol, m);
  else if(method==6)
    solve_gmres_rightpreco(n, x, b, Acsr_apply, invP_apply, niter, tol, m);
  
  /* output true signal and reconstructed one */
  fp= fopen("result.txt", "w");
  fprintf(fp, "x_true \t x_rec \n");
  for(i=0; i<n; i++) fprintf(fp, "%e \t %e\n", xt[i], x[i]);
  fclose(fp);

  Acsr_close();
  free1double(x);
  free1double(b);
  free1double(xt);

  
  return 0;
}

