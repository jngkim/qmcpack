////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by:
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_NUMERIC_MKL_BLAS_LAPACK_H
#define QMCPLUSPLUS_NUMERIC_MKL_BLAS_LAPACK_H

#include "mkl_blas.h"
#include "mkl_cblas.h"
#include "mkl_lapacke.h"

namespace qmcplusplus {
/** Interfaces to blas library using MKL
 *
 *  Arguments (float/double/complex\<float\>/complex\<double\>) determine
 *  which BLAS routines are actually used.
 *  Note that symv can be call in many ways.
 */
inline MKL_Complex16* MM(std::complex<double>* a)
{
  return reinterpret_cast<MKL_Complex16*>(a);
}

inline MKL_Complex8* MM(std::complex<float>* a)
{
  return reinterpret_cast<MKL_Complex8*>(a);
}

inline const MKL_Complex16* MM(const std::complex<double>* a)
{
  return reinterpret_cast<const MKL_Complex16*>(a);
}

inline const MKL_Complex8* MM(const std::complex<float>* a)
{
  return reinterpret_cast<const MKL_Complex8*>(a);
}


// clang-format off
namespace syclBLAS
{
  using index_t=MKL_INT;

  inline  double norm2(index_t n, const double *a, index_t incx = 1)
  {
    return dnrm2(&n, a, &incx);
  }

  inline  double norm2(index_t n, std::complex<double> *a, index_t incx = 1)
  {
    return dznrm2(&n, MM(a), &incx);
  }

  inline  float norm2(index_t n, const float *a, index_t incx = 1)
  {
    return snrm2(&n, a, &incx);
  }

  inline  void scal(index_t n, double alpha, double *x)
  {
    for(int i=0; i<n; ++i) x[i]*=alpha;
  }

  inline  void gemv(char trans_in, index_t n, index_t m, double alpha,
      const double* restrict amat, index_t lda, const double *x,
      index_t incx, double beta, double *y, index_t incy)
  {
    dgemv(&trans_in, &n, &m, &alpha, amat, &lda, x, &incx, &beta, y, &incy);
  }

  inline  void gemv(char trans_in, index_t n, index_t m, float alpha,
      const float *restrict amat, index_t lda, const float *x,
      index_t incx, float beta, float *y, index_t incy)
  {
    sgemv(&trans_in, &n, &m, &alpha, amat, &lda, x, &incx, &beta, y, &incy);
  }

  inline  void gemv(char trans_in, index_t n, index_t m,
      const std::complex<double> &alpha,
      const std::complex<double> *restrict amat, index_t lda,
      const std::complex<double> *restrict x, index_t incx,
      const std::complex<double> &beta,
      std::complex<double> *y, index_t incy)
  {
    zgemv(&trans_in, &n, &m, MM(&alpha), MM(amat), &lda, MM(x), &incx, MM(&beta), MM(y), &incy);
  }

  inline  void gemv(char trans_in, index_t n, index_t m,
      const std::complex<float> &alpha,
      const std::complex<float> *restrict amat, index_t lda,
      const std::complex<float> *restrict x, index_t incx,
      const std::complex<float> &beta,
      std::complex<float> *y, index_t incy)
  {
    cgemv(&trans_in, &n, &m, MM(&alpha), MM(amat), &lda, MM(x), &incx, MM(&beta), MM(y), &incy);
  }

  inline  void gemv(index_t n, index_t m,
                          const std::complex<double> *restrict amat,
                          const std::complex<double> *restrict x,
                          std::complex<double> *restrict y)
  {
    const std::complex<double> done{1.0,0.0};
    const std::complex<double> dzero{};
    gemv('N',m,n,done,amat,m,x,1,dzero,y,1);
  }


  inline  void gemv(index_t n, index_t m, const double *restrict amat,
                          const double *restrict x, double *restrict y)
  {
    gemv('N',m,n,1.0,amat,m,x,1,0.0,y,1);
  }

  inline  void gemv(index_t n, index_t m, const float *restrict amat,
                          const float *restrict x, float *restrict y)
  {
    gemv('N',m,n,1.0f,amat,m,x,1,0.0f,y,1);
  }

  inline  void gemv_trans(index_t n, index_t m, const double *restrict amat,
                                const double *restrict x, double *restrict y)
  {
    gemv('T',m,n,1.0,amat,m,x,1,0.0,y,1);
  }

  inline  void gemv_trans(index_t n, index_t m, const float *restrict amat,
                                const float *restrict x, float *restrict y)
  {
    gemv('T',m,n,1.0f,amat,m,x,1,0.0f,y,1);
  }

  inline  void gemv_trans(index_t n, index_t m,
                                const std::complex<double> *restrict amat,
                                const std::complex<double> *restrict x,
                                std::complex<double> *restrict y)
  {
    gemv('T',m,n,{1.0,0.0},amat,m,x,1,{0.0,0.0},y,1);
  }

  inline  void gemv_trans(index_t n, index_t m,
                                const std::complex<float> *restrict amat,
                                const std::complex<float> *restrict x,
                                std::complex<float> *restrict y)
  {
    gemv('T',m,n,{1.0f,0.0f},amat,m,x,1,{0.0f,0.0f},y,1);
  }

  inline  void gemv(char trans_in, index_t n, index_t m,
                          const std::complex<double> &alpha,
                          const double *restrict amat, index_t lda,
                          const std::complex<double> *restrict x, index_t incx,
                          const std::complex<double> &beta,
                          std::complex<double> *y, index_t incy)
  {
    dzgemv(&trans_in, &n, &m, MM(&alpha), amat, &lda, MM(x), &incx, MM(&beta), MM(y), &incy);
  }

  inline  void gemv(char trans_in, index_t n, index_t m,
                          const std::complex<float> &alpha,
                          const float *restrict amat, index_t lda,
                          const std::complex<float> *restrict x, index_t incx,
                          const std::complex<float> &beta,
                          std::complex<float> *y, index_t incy)
  {
    scgemv(&trans_in, &n, &m, MM(&alpha), amat, &lda, MM(x), &incx, MM(&beta), MM(y), &incy);
  }

  inline  void gemm(char Atrans, char Btrans, index_t M, index_t N, index_t K,
                          double alpha, const double *A, index_t lda,
                          const double *restrict B, index_t ldb, double beta,
                          double *restrict C, index_t ldc)
  {
    dgemm(&Atrans, &Btrans, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }

  inline  void gemm(char Atrans, char Btrans, index_t M, index_t N, index_t K,
                          float alpha, const float *A, index_t lda,
                          const float *restrict B, index_t ldb, float beta,
                          float *restrict C, index_t ldc)
  {
    sgemm(&Atrans, &Btrans, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }

  inline  void gemm(char Atrans, char Btrans, index_t M, index_t N, index_t K,
                          std::complex<double> alpha,
                          const std::complex<double> *A, index_t lda,
                          const std::complex<double> *restrict B, index_t ldb,
                          std::complex<double> beta,
                          std::complex<double> *restrict C, index_t ldc)
  {
    zgemm(&Atrans, &Btrans, &M, &N, &K, MM(&alpha), MM(A), &lda, MM(B), &ldb, MM(&beta), MM(C), &ldc);
  }

  inline  void gemm(char Atrans, char Btrans, index_t M, index_t N, index_t K,
                          std::complex<float> alpha,
                          const std::complex<float> *A, index_t lda,
                          const std::complex<float> *restrict B, index_t ldb,
                          std::complex<float> beta,
                          std::complex<float> *restrict C, index_t ldc)
  {
    cgemm(&Atrans, &Btrans, &M, &N, &K, MM(&alpha), MM(A), &lda, MM(B), &ldb, MM(&beta), MM(C), &ldc);
  }

  ///MKL C interfaces
  inline  void gemv(CBLAS_TRANSPOSE trans_in, index_t m, index_t n, 
      double alpha, const double* restrict amat, index_t lda, 
      const double *x, index_t incx, double beta, double *y, index_t incy)
  {
    cblas_dgemv(CblasColMajor,trans_in,m,n,alpha,amat,lda,x,incx,beta,y,incy);
  }

  inline  void gemv(CBLAS_TRANSPOSE trans_in, index_t m, index_t n, 
      float alpha, const float *restrict amat, index_t lda, 
      const float *x, index_t incx, float beta, float *y, index_t incy)
  {
    cblas_sgemv(CblasColMajor, trans_in,m,n,alpha,amat,lda,x,incx,beta,y,incy);
  }

  inline  void gemv(CBLAS_TRANSPOSE trans_in, index_t m, index_t n, 
      const std::complex<double> &alpha,
      const std::complex<double> *restrict amat, index_t lda,
      const std::complex<double> *restrict x, index_t incx,
      const std::complex<double> &beta,
      std::complex<double> *y, index_t incy)
  {
    cblas_zgemv(CblasColMajor,trans_in,m,n,&alpha,amat,lda,x,incx,&beta,y,incy);
  }

  inline  void gemv(CBLAS_TRANSPOSE trans_in, index_t m, index_t n, 
      const std::complex<float> &alpha,
      const std::complex<float> *restrict amat, index_t lda,
      const std::complex<float> *restrict x, index_t incx,
      const std::complex<float> &beta,
      std::complex<float> *y, index_t incy)
  {
    cblas_cgemv(CblasColMajor, trans_in,m,n,&alpha,amat,lda,x,incx,&beta,y,incy);
  }

  inline  void gemm(CBLAS_TRANSPOSE Atrans, CBLAS_TRANSPOSE Btrans, index_t M, index_t N, index_t K,
                          double alpha, const double *A, index_t lda,
                          const double *restrict B, index_t ldb, double beta,
                          double *restrict C, index_t ldc)
  {
    cblas_dgemm(CblasColMajor, Atrans, Btrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline  void gemm(CBLAS_TRANSPOSE Atrans, CBLAS_TRANSPOSE Btrans, index_t M, index_t N, index_t K,
                          float alpha, const float *A, index_t lda,
                          const float *restrict B, index_t ldb, float beta,
                          float *restrict C, index_t ldc)
  {
    cblas_sgemm(CblasColMajor, Atrans, Btrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline  void gemm(CBLAS_TRANSPOSE Atrans, CBLAS_TRANSPOSE Btrans, index_t M, index_t N, index_t K,
      std::complex<double> alpha,
      const std::complex<double> *A, index_t lda,
      const std::complex<double> *restrict B, index_t ldb,
      std::complex<double> beta,
      std::complex<double> *restrict C, index_t ldc)
  {
    cblas_zgemm(CblasColMajor, Atrans, Btrans, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  inline  void gemm(CBLAS_TRANSPOSE Atrans, CBLAS_TRANSPOSE Btrans, index_t M, index_t N, index_t K,
      std::complex<float> alpha,
      const std::complex<float> *A, index_t lda,
      const std::complex<float> *restrict B, index_t ldb,
      std::complex<float> beta,
      std::complex<float> *restrict C, index_t ldc)
  {
    cblas_cgemm(CblasColMajor, Atrans, Btrans, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  template <typename T>
  inline  T dot(index_t n, const T *restrict a, const T *restrict b)
  {
    T res{};
    for (index_t i = 0; i < n; ++i)
      res += a[i] * b[i];
    return res;
  }

  template <typename T>
  inline  std::complex<T> dot(index_t n, const std::complex<T> *restrict a,
                                    const T *restrict b)
  {
    std::complex<T> res{};
    for (index_t i = 0; i < n; ++i)
      res += a[i] * b[i];
    return res;
  }

  template <typename T>
  inline  std::complex<T> dot(index_t n, const T *restrict a,
                                    const std::complex<T> *restrict b)
  {
    return dot(n,b,a);
  }

  template <typename T>
  inline  void copy(index_t n, const T *restrict a, T *restrict b)
  {
    memcpy(b, a, sizeof(T) * n);
  }

  /** copy using memcpy(target,source,size)
   * @param target starting address of the targe
   * @param source starting address of the source
   * @param number of elements to copy
   */
  template <typename T>
  inline  void copy(T *restrict target, const T *restrict source, index_t n)
  {
    memcpy(target, source, sizeof(T) * n);
  }

  template <typename T>
  inline  void copy(index_t n, const std::complex<T> *restrict a,
                          T *restrict b)
  {
    for (index_t i = 0; i < n; ++i)
      b[i]     = a[i].real();
  }

  template <typename T>
  inline  void copy(index_t n, const T *restrict a,
                          std::complex<T> *restrict b)
  {
    for (index_t i = 0; i < n; ++i)
      b[i]     = a[i];
  }

  template <typename T>
  inline  void copy(index_t n, const T *restrict x, index_t incx, T *restrict y,
                          index_t incy)
  {
    const index_t xmax = incx * n;
    for (index_t ic = 0, jc = 0; ic < xmax; ic += incx, jc += incy)
      y[jc] = x[ic];
  }

  inline  void ger(index_t m, index_t n, double alpha, const double *x, index_t incx,
                         const double *y, index_t incy, double *a, index_t lda)
  {
    dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  }

  inline  void ger(index_t m, index_t n, float alpha, const float *x, index_t incx,
                         const float *y, index_t incy, float *a, index_t lda)
  {
    sger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  }

  inline  void ger(index_t m, index_t n, const std::complex<double> &alpha,
                         const std::complex<double> *x, index_t incx,
                         const std::complex<double> *y, index_t incy,
                         std::complex<double> *a, index_t lda)
  {
    zgeru(&m, &n, MM(&alpha), MM(x), &incx, MM(y), &incy, MM(a), &lda);
  }

  inline  void ger(index_t m, index_t n, const std::complex<float> &alpha,
                         const std::complex<float> *x, index_t incx,
                         const std::complex<float> *y, index_t incy,
                         std::complex<float> *a, index_t lda)
  {
    cgeru(&m, &n, MM(&alpha), MM(x), &incx, MM(y), &incy, MM(a), &lda);
  }
}

namespace LAPACK
{
  using index_t=MKL_INT;

  inline void getrf(index_t n, index_t m, float* a, index_t lda, index_t* piv, index_t& status)
  {
    status=LAPACKE_sgetrf(LAPACK_COL_MAJOR,n,m,a,lda,piv);
  }

  inline void getrf(index_t n, index_t m, std::complex<float>* a, index_t lda, index_t* piv, index_t& status)
  {
    status=LAPACKE_cgetrf(LAPACK_COL_MAJOR,n,m,MM(a),lda,piv);
  }

  inline void getrf(index_t n, index_t m, double* a, index_t lda, index_t* piv, index_t& status)
  {
    status=LAPACKE_dgetrf(LAPACK_COL_MAJOR,n,m,a,lda,piv);
  }

  inline void getrf(index_t n, index_t m, std::complex<double>* a, index_t lda, index_t* piv, index_t& status)
  {
    status=LAPACKE_zgetrf(LAPACK_COL_MAJOR,n,m,MM(a),lda,piv);
  }

  inline void getri(index_t n, float* a, index_t lda, index_t* piv, float* work, index_t& lwork, index_t& status)
  {
    status=LAPACKE_sgetri_work(LAPACK_COL_MAJOR,n,a,lda,piv,work,lwork);
  }

  inline void getri(index_t n, std::complex<float>* a, index_t lda, index_t* piv, std::complex<float>* work, index_t& lwork, index_t& status)
  {
    status=LAPACKE_cgetri_work(LAPACK_COL_MAJOR,n,MM(a),lda,piv,MM(work),lwork);
  }

  inline void getri(index_t n, double* a, index_t lda, index_t* piv, double* work, index_t& lwork, index_t& status)
  {
    status=LAPACKE_dgetri_work(LAPACK_COL_MAJOR,n,a,lda,piv,work,lwork);
  }

  inline void getri(index_t n, std::complex<double>* a, index_t lda, index_t* piv, std::complex<double>* work, index_t& lwork, index_t& status)
  {
    status=LAPACKE_zgetri_work(LAPACK_COL_MAJOR,n,MM(a),lda,piv,MM(work),lwork);
  }

  inline double xlange( const char ochar, index_t m, index_t n, const double* a, index_t lda, double* work )
  {
    return LAPACKE_dlange_work(LAPACK_COL_MAJOR,ochar,m,n,a,lda,work);
  }

  inline float xlange( const char ochar, index_t m, index_t n, const float* a, index_t lda, float* work )
  {
    return LAPACKE_slange_work(LAPACK_COL_MAJOR,ochar,m,n,a,lda,work);
  }

}
}

// clang-format on
#endif // OHMMS_BLAS_H
