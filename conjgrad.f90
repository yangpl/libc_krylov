! /* Demo for linear conjugate gradient method
!  *------------------------------------------------------------------------
!  *
!  * Copyright (c) 2020-2022 Harbin Institute of Technology. All rights reserved.
!  * Author: Pengliang Yang 
!  * Email: ypl.2100@gmail.com
!  * Homepage: https://yangpl.wordpress.com
!  *-----------------------------------------------------------------------*/
program main
  implicit none

  integer :: i, j, n
  real, dimension(:,:), allocatable :: A
  real, dimension(:), allocatable :: x,b

  n = 4
  allocate(x(n))
  allocate(b(n))
  allocate(A(n,n))

  
  !print *, n
  do i=1,n
     x(i) = i*1.0
  enddo

  do i=1,n
     b(i) = 0
     do j=1,n
        A(j,i) = abs(i-j)
        b(i) = b(i) + A(j,i)*x(j)
     enddo
     print *, b(i)
  enddo

  call conjgrad(n, A, b, x)


  do i=1,n
     write(8,*) x(i)
  enddo  
  
  deallocate(x)
  deallocate(b)
  deallocate(A)
  
end program main

!==============================================================
subroutine conjgrad(n, A, b, x)
  implicit none

  integer :: n
  real, dimension(n,n) :: A
  real, dimension(n) :: x,b

  integer ::i,j,iter,niter
  real, dimension(:), allocatable :: p,r,Ap
  real :: alpha, beta, pAp, rsold, rsnew
  
  niter = 30
  
  allocate(p(n))
  allocate(r(n))
  allocate(Ap(n))
  
  do i=1,n
     x(i) = 0
     r(i) = b(i)
     p(i) = r(i)
  enddo
  rsold = dot_product(r,r) !sum(r*r)

  do iter=1,niter
     print *, iter

     do i=1,n
        Ap(i) = dot_product(A(:, i), p)
     enddo
     pAp = dot_product(p, Ap)
     alpha = rsold/pAp
     x(:) = x(:) + alpha*p(:)
     r(:) = r(:) - alpha*Ap(:)
     rsnew = dot_product(r,r)
     if(rsnew < 1e-7) exit

     beta = rsnew/rsold
     p(:) = r(:) + beta*p(:)
     rsold = rsnew
  enddo

  
  deallocate(p)
  deallocate(r)
  deallocate(Ap)
end subroutine conjgrad
