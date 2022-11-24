/* Demo for Krylov space linear optimization algorithms with/without preconditioning
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

//compute dot product s=<x,y>
double ddotprod(int n, double *a, double *b)
{
  int i;
  double s;
  
  for(i=0, s=0; i<n; i++) s += a[i]*b[i];
  return s;  
}

//linear solver using conjugate gradient method
void solve_cg(int n, double *x, double *b, op_t Aop, int niter, double tol)
{
  int i, k;
  double rsold, rsnew, rs0, pAp, alp, beta;

  double *r = alloc1double(n);
  double *p = alloc1double(n);
  double *Ap = alloc1double(n);

  Aop(n, x, Ap);//Ap=Ax0
  for(i=0; i<n; i++) {
    r[i] = b[i]-Ap[i];//r=b-Ax
    p[i] = r[i];
  }
  rsold = ddotprod(n, r, r);
  rs0 = rsold;
  if(rsold<1e-15) return;//residual small enough, exit

  for(k=0; k<niter; k++){
    printf("k=%d rs=%e\n", k, rsold);

    Aop(n, p, Ap);//Ap=A*p
    pAp = ddotprod(n, p, Ap);
    alp = rsold/pAp;

    for(i=0; i<n; i++){
      x[i] += alp*p[i];
      r[i] -= alp*Ap[i];
    }
    rsnew = ddotprod(n, r, r);
    if(rsnew<tol*rs0) {
      printf("converged at k=%d\n", k);
      break;
    }
    
    beta = rsnew/rsold;
    for(i=0; i<n; i++) p[i] = r[i] + beta*p[i];
    rsold = rsnew;
  }

  free1double(r);
  free1double(p);
  free1double(Ap);
}

//linear solver using conjugate gradient method
void solve_pcg(int n, double *x, double *b, op_t Aop, op_t invMop, int niter, double tol)
{
  int i, k;
  double rzold, rznew, rs0, pAp, alp, beta, rs;

  double *r = alloc1double(n);
  double *p = alloc1double(n);
  double *Ap = alloc1double(n);
  double *z = alloc1double(n);

  Aop(n, x, Ap);//Ap=Ax0
  for(i=0; i<n; i++) r[i] = b[i]-Ap[i];//r=b-Ax
  invMop(n, r, z);
  memcpy(p, z, n*sizeof(double));
  rzold = ddotprod(n, r, z);
  rs = ddotprod(n, r, r);
  rs = sqrt(rs);
  rs0 = rs;

  for(k=0; k<niter; k++){
    printf("k=%d rs=%e\n", k, rs);

    Aop(n, p, Ap);//Ap=A*p
    pAp = ddotprod(n, p, Ap);
    alp = rzold/pAp;

    for(i=0; i<n; i++){
      x[i] += alp*p[i];
      r[i] -= alp*Ap[i];
    }
    invMop(n, r, z);
    rznew = ddotprod(n, r, z);
    rs = ddotprod(n, r, r);
    rs = sqrt(rs);
    if(rs<tol*rs0) {
      printf("converged at k=%d\n", k);
      break;
    }
    
    beta = rznew/rzold;
    for(i=0; i<n; i++) p[i] = z[i] + beta*p[i];
    rzold = rznew;
  }

  free1double(r);
  free1double(p);
  free1double(Ap);
  free1double(z);
}

//linear solver using BiCGStab, algorithm 7.7 in Saad book
void solve_bicgstab(int n, double *x, double *b, op_t Aop, int niter, double tol)
{
  int i, k;
  double rs0, rs, rho_old, rho_new, alpha, beta, omega;

  double *r = alloc1double(n);
  double *r0 = alloc1double(n);//rprime0
  double *p = alloc1double(n);
  double *v = alloc1double(n);
  double *s = alloc1double(n);
  double *t = alloc1double(n);
  
  Aop(n, x, v);//v=Ax
  for(i=0; i<n; i++) {
    r[i] = b[i]-v[i];//r=b-Ax
    p[i] = r[i];
    r0[i] = r[i];
  }
  rho_old = ddotprod(n, r0, r);
  rs = ddotprod(n, r, r);
  rs0 = rs;
  
  for(k=0; k<niter; k++){
    printf("k=%d rs=%e\n", k, rs);

    Aop(n, p, v);//v=Ap
    alpha = rho_old/ddotprod(n, r0, v);
    for(i=0; i<n; i++) s[i] = r[i] - alpha*v[i];

    Aop(n, s, t);//t=As
    omega = ddotprod(n, t, s)/ddotprod(n, t, t);

    for(i=0; i<n; i++){
      x[i] += alpha*p[i] + omega*s[i];
      r[i] = s[i] - omega*t[i];
    }
    rs = ddotprod(n, r, r);
    if(rs<tol*rs0) {
      printf("converged at k=%d\n", k);
      break;
    }

    rho_new = ddotprod(n, r0, r);
    beta = (rho_new/rho_old)*alpha/omega;
    for(i=0; i<n; i++) p[i] = r[i] + beta*(p[i]-omega*v[i]);

    rho_old = rho_new;
  }

  free1double(r);
  free1double(r0);
  free1double(p);
  free1double(v);
  free1double(s);
  free1double(t);
}

//Reference: Jie Chen, 2016 JSC, Right preconditioned/Flexible BiCGStab paper
void solve_bicgstab_rightpreco(int n, double *x, double *b, op_t Aop, op_t invMop, int niter, double tol)
{
  int i, k;
  double rs0, rs, rho_old, rho_new, alpha, beta, omega;

  double *r = alloc1double(n);
  double *r0 = alloc1double(n);//rprime0
  double *p = alloc1double(n);
  double *q = alloc1double(n);
  double *v = alloc1double(n);
  double *s = alloc1double(n);
  double *t = alloc1double(n);
  
  Aop(n, x, v);//v=Ax
  for(i=0; i<n; i++) {
    r[i] = b[i]-v[i];//r=b-Ax
    p[i] = r[i];
    r0[i] = r[i];
  }
  rho_old = ddotprod(n, r0, r);
  rs = ddotprod(n, r, r);
  rs0 = rs;
  
  for(k=0; k<niter; k++){
    printf("k=%d rs=%e\n", k, rs);

    invMop(n, p, q);//q=invM*p
    Aop(n, q, v);//v=Aq
    alpha = rho_old/ddotprod(n, r0, v);
    for(i=0; i<n; i++) s[i] = r[i] - alpha*v[i];

    invMop(n, s, r);//r=invM*s
    Aop(n, r, t);//t=Ar
    omega = ddotprod(n, t, s)/ddotprod(n, t, t);

    for(i=0; i<n; i++){
      x[i] += alpha*p[i] + omega*s[i];
      r[i] = s[i] - omega*t[i];
    }
    rs = ddotprod(n, r, r);
    if(rs<tol*rs0) {
      printf("converged at k=%d\n", k);
      break;
    }

    rho_new = ddotprod(n, r0, r);
    beta = (rho_new/rho_old)*alpha/omega;
    for(i=0; i<n; i++) p[i] = r[i] + beta*(p[i]-omega*v[i]);

    rho_old = rho_new;
  }

  free1double(r);
  free1double(r0);
  free1double(p);
  free1double(v);
  free1double(s);
  free1double(t);
}

//linear solver using BiCGStab
void solve_bicgstab0(int n, double *x, double *b, op_t Aop, int niter, double tol)
{
  int i, k;
  double rs0, rs, rho_old, rho_new, alpha, beta, omega;

  double *r = alloc1double(n);
  double *r0 = alloc1double(n);//rprime0
  double *p = alloc1double(n);
  double *v = alloc1double(n);
  double *s = alloc1double(n);
  double *t = alloc1double(n);
  
  Aop(n, x, v);//v=Ax
  for(i=0; i<n; i++) {
    r[i] = b[i]-v[i];//r=b-Ax
    r0[i] = r[i];
  }
  rho_old = 1.;
  alpha = 1.;
  omega = 1.;
  memset(p, 0, n*sizeof(double));
  memset(v, 0, n*sizeof(double));
  rs = ddotprod(n, r, r);
  rs0 = rs;
  
  for(k=0; k<niter; k++){
    printf("k=%d rs=%e\n", k, rs);

    rho_new = ddotprod(n, r0, r);
    beta = (rho_new/rho_old)*alpha/omega;
    for(i=0; i<n; i++) p[i] = r[i] + beta*(p[i]-omega*v[i]);

    Aop(n, p, v);
    alpha = rho_new/ddotprod(n, r0, v);
    for(i=0; i<n; i++) s[i] = r[i] - alpha*v[i];

    Aop(n, s, t);
    omega = ddotprod(n, t, s)/ddotprod(n, t, t);

    for(i=0; i<n; i++){
      x[i] += alpha*p[i] + omega*s[i];
      r[i] = s[i] - omega*t[i];
    }
    rs = ddotprod(n, r, r);
    if(rs<tol*rs0) {
      printf("converged at k=%d\n", k);
      break;
    }
    rho_old = rho_new;
  }

  free1double(r);
  free1double(r0);
  free1double(p);
  free1double(v);
  free1double(s);
  free1double(t);
}

//linear solver using preconditioned BiCGStab
void solve_pbicgstab1(int n, double *x, double *b, op_t Aop, op_t invKop, int niter, double tol)
{
  int i, k;
  double rs0, rs, rho_old, rho_new, alpha, beta;

  double *r = alloc1double(n);
  double *r0 = alloc1double(n);
  double *p = alloc1double(n);
  double *q = alloc1double(n);
  double *u = alloc1double(n);
  double *v = alloc1double(n);
  double *y = alloc1double(n);
  
  Aop(n, x, v);//v=Ax
  for(i=0; i<n; i++) {
    r[i] = b[i]-v[i];//r=b-Ax
    r0[i] = r[i];
  }
  rho_old = 1.;
  memset(p, 0, n*sizeof(double));
  memset(q, 0, n*sizeof(double));
  rs = ddotprod(n, r, r);
  rs0 = rs;
  
  for(k=0; k<niter; k++){
    printf("k=%d rs=%e\n", k, rs);

    rho_new = ddotprod(n, r0, r);
    beta = rho_new/rho_old;
    for(i=0; i<n; i++) {
      u[i] = r[i] + beta*q[i];
      p[i] = u[i] + beta*(q[i]+beta*p[i]);
    }

    invKop(n, p, y);//solve Ky=p, invert K such that y=K^{-1}p
    
    Aop(n, y, v);//v=Ay
    alpha = rho_new/ddotprod(n, r0, v);
    for(i=0; i<n; i++) {
      q[i] = u[i] - alpha*v[i];
      u[i] += q[i];//reuse u to store u+q
    }

    invKop(n, u, y);//solve Kz=u+q, reuse y to store z

    Aop(n, y, v);//v=Az
    for(i=0; i<n; i++){
      x[i] += alpha*y[i];
      r[i] -= alpha*v[i];
    }
    rs = ddotprod(n, r, r);
    if(rs<tol*rs0) {
      printf("converged at k=%d\n", k);
      break;
    }
    rho_old = rho_new;
  }

  free1double(r);
  free1double(r0);
  free1double(p);
  free1double(q);
  free1double(u);
  free1double(v);
  free1double(y);
}

//preconditioned BiCGStab2
void solve_pbicgstab2(int n, double *x, double *b, op_t Aop, op_t invMop, int niter, double tol)
{
  int i, k;
  double rs0, rs, rho_old, rho_new, alpha, beta, gamma;

  double *r = alloc1double(n);
  double *r0 = alloc1double(n);
  double *p = alloc1double(n);
  double *q = alloc1double(n);
  double *s = alloc1double(n);
  double *v = alloc1double(n);
  double *w = alloc1double(n);
  
  Aop(n, x, v);//v=Ax
  for(i=0; i<n; i++) {
    r[i] = b[i]-v[i];//r=b-Ax
    r0[i] = r[i];
    p[i] = r[i];
  }
  rho_old = 1.;
  rs = ddotprod(n, r, r);
  rs0 = rs;
  
  for(k=0; k<niter; k++){
    printf("k=%d rs=%e\n", k, rs);

    invMop(n, p, q);
    Aop(n, q, v);
    rho_new = ddotprod(n, r0, r);
    alpha = rho_new/ddotprod(n, r0, v);
    for(i=0; i<n; i++) {
      x[i] = x[i] + alpha*p[i];
      s[i] = r[i] - alpha*v[i];
    }
    invMop(n, s, q);
    Aop(n, q, w);
    gamma = ddotprod(n, w, s)/ddotprod(n, w, w);

    for(i=0; i<n; i++) {
      x[i] = x[i] + gamma*q[i];
      r[i] = s[i] - gamma*w[i];
    }
    
    beta = (alpha/gamma)*rho_new/rho_old;    
    for(i=0; i<n; i++) {
      p[i] = r[i] + beta*(p[i] - gamma*v[i]);
    }
    rho_old = rho_new;
    
    Aop(n, x, v);//v=Az
    for(i=0; i<n; i++) w[i] = b[i] - v[i];
    rs = ddotprod(n, w, w);
    if(sqrt(rs)<tol*sqrt(rs0)) {
      printf("converged at k=%d\n", k);
      break;
    }
  }

  free1double(r);
  free1double(r0);
  free1double(p);
  free1double(q);
  free1double(s);
  free1double(v);
  free1double(w);
}

//GMRES without preconditioning
void solve_gmres(int n, double *x, double *b, op_t Aop, int niter, double tol, int m)
{
  int i, j, k, iter;

  double *w = alloc1double(n);
  double *r = alloc1double(n);
  double **v = alloc2double(n, m+1);
  double **h = alloc2double(m, m+1);
  double *g = alloc1double(m+1);
  double *y = alloc1double(m+1);
  double *c = alloc1double(m+1);
  double *s = alloc1double(m+1);
  double eps = 1e-15;
  double beta, tmp, r0;

  for(iter=0; iter<niter; iter++){
    /*---------------------------------------------------*/
    Aop(n, x, w);//w=A*x
    for(i=0; i<n; i++) r[i] = b[i] -w[i];
    beta = sqrt(ddotprod(n, r, r));
    if(iter==0) r0 = beta;
    if(beta<tol*r0) return;
    for(i=0; i<n; i++) v[0][i] = r[i]/beta;
    memset(g, 0, (m+1)*sizeof(double));
    g[0] = beta;
    memset(&h[0][0], 0, (m+1)*m*sizeof(double));
    printf("iter=%d error=%e\n", iter, beta);
 
    for(j=0; j<m; j++){
      Aop(n, v[j], w);//r=Av;
      for(i=0; i<=j; i++) {
	h[i][j] = ddotprod(n, v[i], w);
	for(k=0; k<n; k++) w[k] -= h[i][j]*v[i][k];
      }
      h[j+1][j] = sqrt(ddotprod(n, w, w));

      if(h[j+1][j]<eps) { m=j+1; break; }
      for(i=0; i<n; i++) v[j+1][i] = w[i]/h[j+1][j];

      //solve least-squares problem by QR factorization using Given rotations
      //min \|g - H(1:m,1:m) y\|^2, g=(beta,0,...,0)
      //min \|G*g - G*Hy\|^2, G=G(i-1,i,theta)*...*G(2,3,theta)*G(1,2,theta)
      if(j>0){
	for(i=0; i<j; i++){
	  //apply G12, G23,..., G_{j-1,j} to the last column of H_{j,*}
	  tmp = c[i]*h[i][j] + s[i]*h[i+1][j];
	  h[i+1][j] = -s[i]*h[i][j] + c[i]*h[i+1][j];
	  h[i][j] = tmp;
	}
      }

      //compute c=cos(theta) and s=sin(theta)
      if(fabs(h[j][j])>fabs(h[j+1][j])){
	tmp = h[j+1][j]/h[j][j];
	c[j] = 1./sqrt(1. + tmp*tmp);
	s[j] = c[j]*tmp;
      }else{
	tmp = h[j][j]/h[j+1][j];
	s[j] = 1./sqrt(1. + tmp*tmp);
	c[j] = s[j]*tmp;
      }
      h[j][j] = c[j]*h[j][j] + s[j]*h[j+1][j];
      h[j+1][j] = 0.;
      //g=G(j,j+1,theta)g with g[j+1]=0
      g[j+1] = -s[j]*g[j];
      g[j] = c[j]*g[j];

      tmp = fabs(g[j+1]);
      if(tmp<tol*beta) { m=j+1; break; }
      //else printf("j=%d error=%e\n", j, tmp); 
    }

    //now, H becomes an upper triangule matrix, problem min\|g-Hy\|^2 is g=Hy
    //solve it by backward substitution, y = H(1:m,1:m)\g(1:m)
    y[m-1] = g[m-1]/h[m-1][m-1];
    for(i=m-2; i>=0; i--){
      y[i] = g[i];
      for(j=i+1; j<m; j++) y[i] -= h[i][j]*y[j];
      y[i] /= h[i][i];
    }

    //x=x0+Vm*y
    for(j=0; j<m; j++){
      for(i=0; i<n; i++){
	x[i] += y[j]*v[j][i];
      }
    }      
  }//end for iter
  
  free1double(r);
  free1double(w);
  free2double(v);
  free2double(h);
  free1double(g);
  free1double(y);
  free1double(c);
  free1double(s);
}


//GMRES with left preconditioning
void solve_gmres_leftpreco(int n, double *x, double *b, op_t Aop, op_t invMop, int niter, double tol, int m)
{
  int i, j, k, iter;

  double *w = alloc1double(n);
  double *r = alloc1double(n);
  double **v = alloc2double(n, m+1);
  double **h = alloc2double(m, m+1);
  double *g = alloc1double(m+1);
  double *y = alloc1double(m+1);
  double *c = alloc1double(m+1);
  double *s = alloc1double(m+1);
  double eps = 1e-15;
  double beta, tmp, r0;

  for(iter=0; iter<niter; iter++){
    /*-------------------------------------------------------------*/
    Aop(n, x, r);//r=A*x
    for(i=0; i<n; i++) w[i] = b[i] -r[i];
    invMop(n, w, r);//r=invM*(b-Ax)
    beta = sqrt(ddotprod(n, r, r));
    if(iter==0) r0 = beta;
    if(beta<tol*r0) return;
    for(i=0; i<n; i++) v[0][i] = r[i]/beta;
    memset(g, 0, (m+1)*sizeof(double));
    g[0] = beta;
    memset(&h[0][0], 0, (m+1)*m*sizeof(double));
    printf("iter=%d error=%e\n", iter, beta);
 
    for(j=0; j<m; j++){
      Aop(n, v[j], r);//r=Av;
      invMop(n, r, w);//w=invM*Av;
      for(i=0; i<=j; i++) {
	h[i][j] = ddotprod(n, v[i], w);
	for(k=0; k<n; k++) w[k] -= h[i][j]*v[i][k];
      }
      h[j+1][j] = sqrt(ddotprod(n, w, w));

      if(h[j+1][j]<eps) { m=j+1; break; }
      for(i=0; i<n; i++) v[j+1][i] = w[i]/h[j+1][j];

      //solve least-squares problem by QR factorization using Given rotations
      //min \|g - H(1:m,1:m) y\|^2, g=(beta,0,...,0)
      //min \|G*g - G*Hy\|^2, G=G(i-1,i,theta)*...*G(2,3,theta)*G(1,2,theta)
      if(j>0){
	for(i=0; i<j; i++){
	  //apply G12, G23,..., G_{j-1,j} to the last column of H_{j,*}
	  tmp = c[i]*h[i][j] + s[i]*h[i+1][j];
	  h[i+1][j] = -s[i]*h[i][j] + c[i]*h[i+1][j];
	  h[i][j] = tmp;
	}
      }

      //compute c=cos(theta) and s=sin(theta)
      if(fabs(h[j][j])>fabs(h[j+1][j])){
	tmp = h[j+1][j]/h[j][j];
	c[j] = 1./sqrt(1. + tmp*tmp);
	s[j] = c[j]*tmp;
      }else{
	tmp = h[j][j]/h[j+1][j];
	s[j] = 1./sqrt(1. + tmp*tmp);
	c[j] = s[j]*tmp;
      }
      h[j][j] = c[j]*h[j][j] + s[j]*h[j+1][j];
      h[j+1][j] = 0.;
      //g=G(j,j+1,theta)g with g[j+1]=0
      g[j+1] = -s[j]*g[j];
      g[j] = c[j]*g[j];

      tmp = fabs(g[j+1]);
      if(tmp<tol*beta) { m = j+1; break; }
      //else printf("j=%d error=%e\n", j, tmp); 
    }

    //now, H becomes an upper triangule matrix, problem min\|g-Hy\|^2 is g=Hy
    //solve it by backward substitution, y = H(1:m,1:m)\g(1:m)
    y[m-1] = g[m-1]/h[m-1][m-1];
    for(i=m-2; i>=0; i--){
      y[i] = g[i];
      for(j=i+1; j<m; j++) y[i] -= h[i][j]*y[j];
      y[i] /= h[i][i];
    }

    //x=x0+Vm*y
    for(j=0; j<m; j++){
      for(i=0; i<n; i++){
	x[i] += y[j]*v[j][i];
      }
    }
  }//end for iter

  
  free1double(r);
  free1double(w);
  free2double(v);
  free2double(h);
  free1double(g);
  free1double(y);
  free1double(c);
  free1double(s);
}


//GMRES with right preconditioning
void solve_gmres_rightpreco(int n, double *x, double *b, op_t Aop, op_t invMop, int niter, double tol, int m)
{
  int i, j, k, iter;

  double *w = alloc1double(n);
  double *r = alloc1double(n);
  double **v = alloc2double(n, m+1);
  double **h = alloc2double(m, m+1);
  double *g = alloc1double(m+1);
  double *y = alloc1double(m+1);
  double *c = alloc1double(m+1);
  double *s = alloc1double(m+1);
  double eps = 1e-15;
  double beta, tmp, r0;

  for(iter=0; iter<niter; iter++){
    /*---------------------------------------------------*/
    Aop(n, x, w);//w=A*x
    for(i=0; i<n; i++) r[i] = b[i]-w[i];
    beta = sqrt(ddotprod(n, r, r));
    if(iter==0) r0 = beta;
    if(beta<tol*r0) return;
    for(i=0; i<n; i++) v[0][i] = r[i]/beta;
    memset(g, 0, (m+1)*sizeof(double));
    g[0] = beta;
    memset(&h[0][0], 0, (m+1)*m*sizeof(double));
    printf("iter=%d error=%e\n", iter, beta);
 
    for(j=0; j<m; j++){
      invMop(n, v[j], r);//r=invM*v
      Aop(n, r, w);//w=Ar;
      for(i=0; i<=j; i++) {
	h[i][j] = ddotprod(n, v[i], w);
	for(k=0; k<n; k++) w[k] -= h[i][j]*v[i][k];
      }
      h[j+1][j] = sqrt(ddotprod(n, w, w));

      if(h[j+1][j]<eps) { m=j+1; break; }
      for(i=0; i<n; i++) v[j+1][i] = w[i]/h[j+1][j];

      //solve least-squares problem by QR factorization using Given rotations
      //min \|g - H(1:m,1:m) y\|^2, g=(beta,0,...,0)
      //min \|G*g - G*Hy\|^2, G=G(i-1,i,theta)*...*G(2,3,theta)*G(1,2,theta)
      if(j>0){
	for(i=0; i<j; i++){
	  //apply G12, G23,..., G_{j-1,j} to the last column of H_{j,*}
	  tmp = c[i]*h[i][j] + s[i]*h[i+1][j];
	  h[i+1][j] = -s[i]*h[i][j] + c[i]*h[i+1][j];
	  h[i][j] = tmp;
	}
      }

      //compute c=cos(theta) and s=sin(theta)
      if(fabs(h[j][j])>fabs(h[j+1][j])){
	tmp = h[j+1][j]/h[j][j];
	c[j] = 1./sqrt(1. + tmp*tmp);
	s[j] = c[j]*tmp;
      }else{
	tmp = h[j][j]/h[j+1][j];
	s[j] = 1./sqrt(1. + tmp*tmp);
	c[j] = s[j]*tmp;
      }
      h[j][j] = c[j]*h[j][j] + s[j]*h[j+1][j];
      h[j+1][j] = 0.;
      //g=G(j,j+1,theta)g with g[j+1]=0
      g[j+1] = -s[j]*g[j];
      g[j] = c[j]*g[j];

      tmp = fabs(g[j+1]);
      if(tmp<tol*beta) { m=j+1; break; }
      //else printf("j=%d error=%e\n", j, tmp); 
    }

    //now, H becomes an upper triangule matrix, problem min\|g-Hy\|^2 is g=Hy
    //solve it by backward substitution, y = H(1:m,1:m)\g(1:m)
    y[m-1] = g[m-1]/h[m-1][m-1];
    for(i=m-2; i>=0; i--){
      y[i] = g[i];
      for(j=i+1; j<m; j++) y[i] -= h[i][j]*y[j];
      y[i] /= h[i][i];
    }

    for(i=0; i<n; i++) r[i] = 0;
    for(j=0; j<m; j++){
      for(i=0; i<n; i++){
	r[i] += y[j]*v[j][i];//r=Vm*y
      }
    }      
    invMop(n, r, w);//w=invM*Vm*y
    //x=x0+invM*Vm*y
    for(i=0; i<n; i++) x[i] += w[i];
  }//end for iter
  
  free1double(r);
  free1double(w);
  free2double(v);
  free2double(h);
  free1double(g);
  free1double(y);
  free1double(c);
  free1double(s);
}

//CGNR method
void solve_cgnr(int n, double *x, double *b, op_t Aop, op_t Atop, int niter, double tol)
{
  int i, iter;
  double *r, *z, *p, *w;
  double zsold, zsnew, ws, rs, rs0, alpha, beta;
  
  r = alloc1double(n);
  z = alloc1double(n);
  p = alloc1double(n);
  w = alloc1double(n);
  
  Aop(n, x, w);//w=A*x
  for(i=0; i<n; i++) r[i] = b[i] - w[i];
  Atop(n, r, z);//z=At*r
  for(i=0; i<n; i++) p[i] = z[i];
  zsold = ddotprod(n, z, z);

  for(iter=0; iter<niter; iter++){
    rs = ddotprod(n, r, r);
    printf("iter=%d cgnr rs=%e\n", iter, rs);
    if(iter==0) rs0 = rs;
    if(rs<tol*rs0) break;
    
    Aop(n, p, w);//w=Ap
    ws = ddotprod(n, w, w);
    alpha = zsold/ws;
    for(i=0; i<n; i++) {
      x[i] += alpha*p[i];
      r[i] -= alpha*w[i];
    }
    Atop(n, r, z);
    zsnew = ddotprod(n, z, z);
    beta = zsnew/zsold;
    for(i=0; i<n; i++) p[i] = z[i] + beta*p[i];
    zsold = zsnew;
  }  

  free1double(r);
  free1double(z);
  free1double(w);
  free1double(p);
}
