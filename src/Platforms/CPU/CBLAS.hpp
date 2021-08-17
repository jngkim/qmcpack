//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_NUMERIC_MKL_H
#define QMCPLUSPLUS_NUMERIC_MKL_H

#include <complex>
#include <iostream>
#include "mkl_cblas.h"
#include "mkl_lapacke.h"

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

/** Interfaces to blas library
 *
 *   static data members to facilitate /Fortran blas interface
 *   static member functions to use blas functions
 *   - inline static void axpy
 *   - inline static double norm2
 *   - inline static float norm2
 *   - inline static void symv
 *   - inline static double dot
 *   - inline static float dot
 *
 *  Arguments (float/double/complex\<float\>/complex\<double\>) determine
 *  which BLAS routines are actually used.
 *  Note that symv can be call in many ways.
 */
namespace BLAS
{
  template<typename T>
  inline static void axpy(int n, T x, const T* restrict a, T* restrict b) { 
    for(int i=0; i<n; ++i) b[i]+=x*a[i];
  }

  template<typename T>
  inline static void axpy(int n, T x, const T* restrict a, int inca, T* restrict b, int incb) { 
    const int nmax=n*inca;
    for(int i=0,j=0; i<nmax; i+=inca,j+=incb) b[i]+=x*a[j];
  }

#if 0
  inline static float norm2(int n, const float* a, int incx = 1) { return snrm2(n, a, incx); }

  inline static float norm2(int n, const std::complex<float>* a, int incx = 1) { return scnrm2(n, a, incx); }

  inline static double norm2(int n, const double* a, int incx = 1) { return dnrm2(n, a, incx); }

  inline static double norm2(int n, const std::complex<double>* a, int incx = 1) { return dznrm2(n, a, incx); }

  inline static void scal(int n, float alpha, float* x, int incx = 1) { sscal(n, alpha, x, incx); }

  inline static void scal(int n, std::complex<float> alpha, std::complex<float>* x, int incx = 1)
  {
    cscal(n, alpha, x, incx);
  }

  inline static void scal(int n, double alpha, double* x, int incx = 1) { dscal(n, alpha, x, incx); }

  inline static void scal(int n, std::complex<double> alpha, std::complex<double>* x, int incx = 1)
  {
    zscal(n, alpha, x, incx);
  }

  inline static void scal(int n, double alpha, std::complex<double>* x, int incx = 1) { zdscal(n, alpha, x, incx); }

  inline static void scal(int n, float alpha, std::complex<float>* x, int incx = 1) { csscal(n, alpha, x, incx); }
#endif

  inline static void gemv(char trans_in,
                          int n,
                          int m,
                          double alpha,
                          const double* restrict amat,
                          int lda,
                          const double* x,
                          int incx,
                          double beta,
                          double* y,
                          int incy)
  {
    const CBLAS_TRANSPOSE trans=(trans_in=='T')? CblasTrans: CblasNoTrans;
    cblas_dgemv(CblasColMajor,trans, n, m, alpha, amat, lda, x, incx, beta, y, incy);
  }

  inline static void gemv(char trans_in,
                          int n,
                          int m,
                          float alpha,
                          const float* restrict amat,
                          int lda,
                          const float* x,
                          int incx,
                          float beta,
                          float* y,
                          int incy)
  {
    const CBLAS_TRANSPOSE trans=(trans_in=='T')? CblasTrans: CblasNoTrans;
    cblas_sgemv(CblasColMajor,trans, n, m, alpha, amat, lda, x, incx, beta, y, incy);
  }

  inline static void gemv(char trans_in,
                          int n,
                          int m,
                          const std::complex<double>& alpha,
                          const std::complex<double>* restrict amat,
                          int lda,
                          const std::complex<double>* restrict x,
                          int incx,
                          const std::complex<double>& beta,
                          std::complex<double>* y,
                          int incy)
  {
    const CBLAS_TRANSPOSE trans=(trans_in=='T')? CblasConjTrans: CblasNoTrans;
    cblas_zgemv(CblasColMajor,trans, n, m, MM(&alpha), MM(amat), lda, MM(x), incx, MM(&beta), MM(y), incy);
  }

  inline static void gemv(char trans_in,
                          int n,
                          int m,
                          const std::complex<float>& alpha,
                          const std::complex<float>* restrict amat,
                          int lda,
                          const std::complex<float>* restrict x,
                          int incx,
                          const std::complex<float>& beta,
                          std::complex<float>* y,
                          int incy)
  {
    const CBLAS_TRANSPOSE trans=(trans_in=='T')? CblasConjTrans: CblasNoTrans;
    cblas_cgemv(CblasColMajor,trans, n, m, MM(&alpha), MM(amat), lda, MM(x), incx, MM(&beta), MM(y), incy);
  }

  template<typename T>
  inline static void gemv(int n, int m, const T* restrict amat, const T* restrict x, T* restrict y)
  {
    constexpr T one_  = 1.0e0;
    constexpr T zero_ = 0.0e0;
    gemv('N', m, n, one_, amat, m, x, 1, zero_, y, 1);
  }


  template<typename T>
  inline static void gemv_trans(int n, int m, const T* restrict amat, const T* restrict x, T* restrict y)
  {
    constexpr T one_   = 1.0e0;
    constexpr T zero_  = 0.0e0;
    gemv('T', m, n, one_, amat, m, x, 1, zero_, y, 1);
  }



  inline static void gemm(char Atrans,
                          char Btrans,
                          int M,
                          int N,
                          int K,
                          double alpha,
                          const double* A,
                          int lda,
                          const double* restrict B,
                          int ldb,
                          double beta,
                          double* restrict C,
                          int ldc)
  {
    const CBLAS_TRANSPOSE tA=(Atrans=='T')? CblasTrans: CblasNoTrans;
    const CBLAS_TRANSPOSE tB=(Btrans=='T')? CblasTrans: CblasNoTrans;
    cblas_dgemm(CblasColMajor,tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline static void gemm(char Atrans,
                          char Btrans,
                          int M,
                          int N,
                          int K,
                          float alpha,
                          const float* A,
                          int lda,
                          const float* restrict B,
                          int ldb,
                          float beta,
                          float* restrict C,
                          int ldc)
  {
    const CBLAS_TRANSPOSE tA=(Atrans=='T')? CblasTrans: CblasNoTrans;
    const CBLAS_TRANSPOSE tB=(Btrans=='T')? CblasTrans: CblasNoTrans;
    cblas_sgemm(CblasColMajor,tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline static void gemm(char Atrans,
                          char Btrans,
                          int M,
                          int N,
                          int K,
                          std::complex<double> alpha,
                          const std::complex<double>* A,
                          int lda,
                          const std::complex<double>* restrict B,
                          int ldb,
                          std::complex<double> beta,
                          std::complex<double>* restrict C,
                          int ldc)
  {
    const CBLAS_TRANSPOSE tA=(Atrans=='T')? CblasTrans: CblasNoTrans;
    const CBLAS_TRANSPOSE tB=(Btrans=='T')? CblasTrans: CblasNoTrans;
    cblas_zgemm(CblasColMajor,tA, tB, M, N, K, MM(&alpha), MM(A), lda, MM(B), ldb, MM(&beta), MM(C), ldc);
  }

  inline static void gemm(char Atrans,
                          char Btrans,
                          int M,
                          int N,
                          int K,
                          std::complex<float> alpha,
                          const std::complex<float>* A,
                          int lda,
                          const std::complex<float>* restrict B,
                          int ldb,
                          std::complex<float> beta,
                          std::complex<float>* restrict C,
                          int ldc)
  {
    const CBLAS_TRANSPOSE tA=(Atrans=='T')? CblasTrans: CblasNoTrans;
    const CBLAS_TRANSPOSE tB=(Btrans=='T')? CblasTrans: CblasNoTrans;
    cblas_cgemm(CblasColMajor,tA, tB, M, N, K, MM(&alpha), MM(A), lda, MM(B), ldb, MM(&beta), MM(C), ldc);
  }

  template<typename T>
  inline static T dot(int n, const T* restrict a, const T* restrict b)
  {
    T res = T{};
    for (int i = 0; i < n; ++i)
      res += a[i] * b[i];
    return res;
  }


  template<typename T>
  inline static T dot(int n, const T* restrict a, int incx, const T* restrict b, int incy)
  {
    T res{};
    for (int i = 0, ia = 0, ib = 0; i < n; ++i, ia += incx, ib += incy)
      res += a[ia] * b[ib];
    return res;
  }

  template<typename T>
  inline static void copy(int n, const T* restrict a, T* restrict b)
  {
    memcpy(b, a, sizeof(T) * n);
  }

  /** copy using memcpy(target,source,size)
   * @param target starting address of the targe
   * @param source starting address of the source
   * @param number of elements to copy
   */
  template<typename T>
  inline static void copy(T* restrict target, const T* restrict source, int n)
  {
    memcpy(target, source, sizeof(T) * n);
  }

  template<typename T>
  inline static void copy(int n, const std::complex<T>* restrict a, T* restrict b)
  {
    for (int i = 0; i < n; ++i)
      b[i] = a[i].real();
  }

  template<typename T>
  inline static void copy(int n, const T* restrict a, std::complex<T>* restrict b)
  {
    for (int i = 0; i < n; ++i)
      b[i] = a[i];
  }

  template<typename T>
  inline static void copy(int n, const T* restrict x, int incx, T* restrict y, int incy)
  {
    const int xmax = incx * n;
    for (int ic = 0, jc = 0; ic < xmax; ic += incx, jc += incy)
      y[jc] = x[ic];
  }

  /*
    inline static
    void copy(int n, double x, double* a) {
      dinit(n,x,a,INCX);
    }

    inline static
    void copy(int n, const std::complex<double>* restrict a, std::complex<double>* restrict b)
    {
      zcopy(n,a,INCX,b,INCY);
    }

    inline static
    void copy(int n, const std::complex<double>* restrict a, int ia, std::complex<double>* restrict b, int ib) {
      zcopy(n,a,ia,b,ib);
    }
  */

  inline static void ger(int m,
                         int n,
                         double alpha,
                         const double* x,
                         int incx,
                         const double* y,
                         int incy,
                         double* a,
                         int lda)
  {
    cblas_dger(CblasColMajor, m, n, alpha, x, incx, y,incy, a, lda);
  }

  inline static void ger(int m,
                         int n,
                         float alpha,
                         const float* x,
                         int incx,
                         const float* y,
                         int incy,
                         float* a,
                         int lda)
  {
    cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda);
  }

  inline static void ger(int m,
                         int n,
                         const std::complex<double>& alpha,
                         const std::complex<double>* x,
                         int incx,
                         const std::complex<double>* y,
                         int incy,
                         std::complex<double>* a,
                         int lda)
  {
    cblas_zgeru(CblasColMajor, m, n, MM(&alpha), MM(x), incx, MM(y), incy, MM(a), lda);
  }

  inline static void ger(int m,
                         int n,
                         const std::complex<float>& alpha,
                         const std::complex<float>* x,
                         int incx,
                         const std::complex<float>* y,
                         int incy,
                         std::complex<float>* a,
                         int lda)
  {
    cblas_cgeru(CblasColMajor, m, n, MM(&alpha), MM(x), incx, MM(y), incy, MM(a), lda);
  }
};

struct LAPACK
{
  inline static void heev(char jobz,
                          char uplo,
                          int n,
                          std::complex<float>* a,
                          int lda,
                          float* w,
                          std::complex<float>* work,
                          int lwork,
                          float* rwork,
                          int& info)
  {
    info=LAPACKE_cheev_work(CblasColMajor,jobz, uplo, n, MM(a), lda, w, MM(work), lwork, rwork);
  }

  inline static void heev(char jobz,
                          char uplo,
                          int n,
                          std::complex<double>* a,
                          int lda,
                          double* w,
                          std::complex<double>* work,
                          int lwork,
                          double* rwork,
                          int& info)
  {
    info=LAPACKE_zheev_work(CblasColMajor,jobz, uplo, n, MM(a), lda, w, MM(work), lwork, rwork);
  }

  inline static void gesvd(const char& jobu,
                           const char& jobvt,
                           const int& m,
                           const int& n,
                           float* a,
                           const int& lda,
                           float* s,
                           float* u,
                           const int& ldu,
                           float* vt,
                           const int& ldvt,
                           float* work,
                           const int& lwork,
                           int& info)
  {
    info = LAPACKE_sgesvd_work(CblasColMajor,jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
  }

  inline static void gesvd(const char& jobu,
                           const char& jobvt,
                           const int& m,
                           const int& n,
                           double* a,
                           const int& lda,
                           double* s,
                           double* u,
                           const int& ldu,
                           double* vt,
                           const int& ldvt,
                           double* work,
                           const int& lwork,
                           int& info)
  {
    info = LAPACKE_dgesvd_work(CblasColMajor,jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
  }

  inline static void gesvd(const char& jobu,
                           const char& jobvt,
                           const int& m,
                           const int& n,
                           std::complex<float>* a,
                           const int& lda,
                           float* s,
                           std::complex<float>* u,
                           const int& ldu,
                           std::complex<float>* vt,
                           const int& ldvt,
                           std::complex<float>* work,
                           const int& lwork,
                           float* rwork,
                           int& info)
  {
    info=LAPACKE_cgesvd_work(CblasColMajor,jobu, jobvt, m, n, MM(a), lda, s, MM(u), ldu, MM(vt), ldvt, MM(work), lwork, rwork);
  }

  inline static void gesvd(const char& jobu,
                           const char& jobvt,
                           const int& m,
                           const int& n,
                           std::complex<double>* a,
                           const int& lda,
                           double* s,
                           std::complex<double>* u,
                           const int& ldu,
                           std::complex<double>* vt,
                           const int& ldvt,
                           std::complex<double>* work,
                           const int& lwork,
                           double* rwork,
                           int& info)
  {
    info=LAPACKE_zgesvd_work(CblasColMajor,jobu, jobvt, m, n, MM(a), lda, s, MM(u), ldu, MM(vt), ldvt, MM(work), lwork, rwork);
  }

  inline static void geev(char* jobvl,
                          char* jobvr,
                          int* n,
                          double* a,
                          int* lda,
                          double* alphar,
                          double* alphai,
                          double* vl,
                          int* ldvl,
                          double* vr,
                          int* ldvr,
                          double* work,
                          int* lwork,
                          int* info)
  {
    *info=LAPACKE_dgeev_work(CblasColMajor,*jobvl, *jobvr, *n, a, *lda, alphar, alphai, vl, *ldvl, vr, *ldvr, work, *lwork);
  }

  inline static void geev(char* jobvl,
                          char* jobvr,
                          int* n,
                          float* a,
                          int* lda,
                          float* alphar,
                          float* alphai,
                          float* vl,
                          int* ldvl,
                          float* vr,
                          int* ldvr,
                          float* work,
                          int* lwork,
                          int* info)
  {
    *info=LAPACKE_sgeev_work(CblasColMajor,*jobvl, *jobvr, *n, a, *lda, alphar, alphai, vl, *ldvl, vr, *ldvr, work, *lwork);
  }

  inline static void ggev(char* jobvl,
                          char* jobvr,
                          int* n,
                          double* a,
                          int* lda,
                          double* b,
                          int* ldb,
                          double* alphar,
                          double* alphai,
                          double* beta,
                          double* vl,
                          int* ldvl,
                          double* vr,
                          int* ldvr,
                          double* work,
                          int* lwork,
                          int* info)
  {
    *info=LAPACKE_dggev_work(CblasColMajor,*jobvl, *jobvr, *n, a, *lda, b, *ldb, alphar, alphai, beta, vl, *ldvl, vr, *ldvr, work, *lwork);
  }

  inline static void ggev(char* jobvl,
                          char* jobvr,
                          int* n,
                          float* a,
                          int* lda,
                          float* b,
                          int* ldb,
                          float* alphar,
                          float* alphai,
                          float* beta,
                          float* vl,
                          int* ldvl,
                          float* vr,
                          int* ldvr,
                          float* work,
                          int* lwork,
                          int* info)
  {
    *info=LAPACKE_sggev_work(CblasColMajor,*jobvl, *jobvr, *n, a, *lda, b, *ldb, alphar, alphai, beta, vl, *ldvl, vr, *ldvr, work, *lwork);
  }

  inline static void hevr(char& JOBZ,
                          char& RANGE,
                          char& UPLO,
                          int& N,
                          float* A,
                          int& LDA,
                          float& VL,
                          float& VU,
                          int& IL,
                          int& IU,
                          float& ABSTOL,
                          lapack_int& M,
                          float* W,
                          float* Z,
                          int& LDZ,
                          lapack_int* ISUPPZ,
                          float* WORK,
                          int& LWORK,
                          float* RWORK,
                          int& LRWORK,
                          lapack_int* IWORK,
                          int& LIWORK,
                          int& INFO)
  {
    if (WORK)
      WORK[0] = 0;
    INFO=LAPACKE_ssyevr_work(CblasColMajor,
        JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL, &M, W, Z, LDZ, ISUPPZ, RWORK, LRWORK, IWORK, LIWORK);
  }

  inline static void hevr(char& JOBZ,
                          char& RANGE,
                          char& UPLO,
                          int& N,
                          double* A,
                          int& LDA,
                          double& VL,
                          double& VU,
                          int& IL,
                          int& IU,
                          double& ABSTOL,
                          lapack_int& M,
                          double* W,
                          double* Z,
                          int& LDZ,
                          lapack_int* ISUPPZ,
                          double* WORK,
                          int& LWORK,
                          double* RWORK,
                          int& LRWORK,
                          lapack_int* IWORK,
                          int& LIWORK,
                          int& INFO)
  {
    if (WORK)
      WORK[0] = 0;
    INFO=LAPACKE_dsyevr_work(CblasColMajor,
        JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL, &M, W, Z, LDZ, ISUPPZ, RWORK, LRWORK, IWORK, LIWORK);
  }

  inline static void hevr(char& JOBZ,
                          char& RANGE,
                          char& UPLO,
                          int& N,
                          std::complex<float>* A,
                          int& LDA,
                          float& VL,
                          float& VU,
                          int& IL,
                          int& IU,
                          float& ABSTOL,
                          lapack_int& M,
                          float* W,
                          std::complex<float>* Z,
                          int& LDZ,
                          lapack_int* ISUPPZ,
                          std::complex<float>* WORK,
                          int& LWORK,
                          float* RWORK,
                          int& LRWORK,
                          lapack_int* IWORK,
                          int& LIWORK,
                          int& INFO)
  {
    INFO=LAPACKE_cheevr_work(CblasColMajor,
        JOBZ, RANGE, UPLO, N, MM(A), LDA, VL, VU, IL, IU, ABSTOL, &M, W, MM(Z), LDZ, ISUPPZ, MM(WORK), LWORK, RWORK, LRWORK,
           IWORK, LIWORK);
  }

  inline static void hevr(char& JOBZ,
                          char& RANGE,
                          char& UPLO,
                          int& N,
                          std::complex<double>* A,
                          int& LDA,
                          double& VL,
                          double& VU,
                          int& IL,
                          int& IU,
                          double& ABSTOL,
                          lapack_int& M,
                          double* W,
                          std::complex<double>* Z,
                          int& LDZ,
                          lapack_int* ISUPPZ,
                          std::complex<double>* WORK,
                          int& LWORK,
                          double* RWORK,
                          int& LRWORK,
                          lapack_int* IWORK,
                          int& LIWORK,
                          int& INFO)
  {
    INFO=LAPACKE_zheevr_work(CblasColMajor,
        JOBZ, RANGE, UPLO, N, MM(A), LDA, VL, VU, IL, IU, ABSTOL, &M, W, MM(Z), LDZ, ISUPPZ, MM(WORK), LWORK, RWORK, LRWORK,
           IWORK, LIWORK);
  }

  static lapack_int  getrf(const int n, const int m, double* a, const int n0, lapack_int* piv)
  {
    return LAPACKE_dgetrf(CblasColMajor, n, m, a, n0, piv);
  }

  static lapack_int getrf(const int n, const int m, float* a, const int n0, lapack_int* piv)
  {
    return LAPACKE_sgetrf(CblasColMajor, n, m, a, n0, piv);
  }

  static lapack_int  getrf(const int n, const int m, std::complex<double>* a, const int n0, lapack_int* piv)
  {
    return LAPACKE_zgetrf(CblasColMajor, n, m, MM(a), n0, piv);
  }

  static lapack_int getrf(const int n, const int m, std::complex<float>* a, const int n0, lapack_int* piv)
  {
    return LAPACKE_cgetrf(CblasColMajor, n, m, MM(a), n0, piv);
  }

  static lapack_int getri(int n,
                    float* restrict a,
                    int n0,
                    lapack_int const* restrict piv,
                    float* restrict work,
                    int const n1)
  {
    return LAPACKE_sgetri_work(CblasColMajor, n, a, n0, piv, work, n1);
  }

  static lapack_int  getri(int n,
                    double* restrict a,
                    int n0,
                    lapack_int const* restrict piv,
                    double* restrict work,
                    int const n1)
  {
    return LAPACKE_dgetri_work(CblasColMajor, n, a, n0, piv, work, n1);
  }

  static lapack_int  getri(int n,
                    std::complex<float>* restrict a,
                    int n0,
                    lapack_int const* restrict piv,
                    std::complex<float>* restrict work,
                    int const n1)
  {
    return LAPACKE_cgetri_work(CblasColMajor, n, MM(a), n0, piv, MM(work), n1);
  }

  static lapack_int getri(int n,
                    std::complex<double>* restrict a,
                    int n0,
                    lapack_int const* restrict piv,
                    std::complex<double>* restrict work,
                    int const n1)
  {
    return LAPACKE_zgetri_work(CblasColMajor, n, MM(a), n0, piv, MM(work), n1);
  }

  static lapack_int  geqrf(int M,
                    int N,
                    std::complex<double>* A,
                    const int LDA,
                    std::complex<double>* TAU,
                    std::complex<double>* WORK,
                    int LWORK)
  {
    return LAPACKE_zgeqrf_work(CblasColMajor, M, N, MM(A), LDA, MM(TAU), MM(WORK), LWORK);
  }

  static void geqrf(int M, int N, double* A, const int LDA, double* TAU, double* WORK, int LWORK, int& INFO)
  {
    INFO=LAPACKE_dgeqrf_work(CblasColMajor, M, N, A, LDA, TAU, WORK, LWORK);
  }

  static void geqrf(int M,
                    int N,
                    std::complex<float>* A,
                    const int LDA,
                    std::complex<float>* TAU,
                    std::complex<float>* WORK,
                    int LWORK,
                    int& INFO)
  {
    INFO=LAPACKE_cgeqrf_work(CblasColMajor, M, N, MM(A), LDA, MM(TAU), MM(WORK), LWORK);
  }

  static void geqrf(int M, int N, float* A, const int LDA, float* TAU, float* WORK, int LWORK, int& INFO)
  {
    INFO=LAPACKE_sgeqrf_work(CblasColMajor, M, N, A, LDA, TAU, WORK, LWORK);
  }

  static void gelqf(int M,
                    int N,
                    std::complex<double>* A,
                    const int LDA,
                    std::complex<double>* TAU,
                    std::complex<double>* WORK,
                    int LWORK,
                    int& INFO)
  {
    INFO=LAPACKE_zgelqf_work(CblasColMajor,M, N, MM(A), LDA, MM(TAU), MM(WORK), LWORK);
  }

  static void gelqf(int M, int N, double* A, const int LDA, double* TAU, double* WORK, int LWORK, int& INFO)
  {
    INFO=LAPACKE_dgelqf_work(CblasColMajor,M, N, A, LDA, TAU, WORK, LWORK);
  }

  static void gelqf(int M,
                    int N,
                    std::complex<float>* A,
                    const int LDA,
                    std::complex<float>* TAU,
                    std::complex<float>* WORK,
                    int LWORK,
                    int& INFO)
  {
    INFO=LAPACKE_cgelqf_work(CblasColMajor,M, N, MM(A), LDA, MM(TAU), MM(WORK), LWORK);
  }

  static void gelqf(int M, int N, float* A, const int LDA, float* TAU, float* WORK, int LWORK, int& INFO)
  {
    INFO=LAPACKE_sgelqf_work(CblasColMajor,M, N, A, LDA, TAU, WORK, LWORK);
  }

  static void gqr(int M,
                  int N,
                  int K,
                  std::complex<double>* A,
                  const int LDA,
                  std::complex<double>* TAU,
                  std::complex<double>* WORK,
                  int LWORK,
                  int& INFO)
  {
    INFO=LAPACKE_zungqr_work(CblasColMajor, M, N, K, MM(A), LDA, MM(TAU), MM(WORK), LWORK);
  }

  static void gqr(int M, int N, int K, double* A, const int LDA, double* TAU, double* WORK, int LWORK, int& INFO)
  {
    INFO=LAPACKE_dorgqr_work(CblasColMajor, M, N, K, A, LDA, TAU, WORK, LWORK);
  }

  static void gqr(int M,
                  int N,
                  int K,
                  std::complex<float>* A,
                  const int LDA,
                  std::complex<float>* TAU,
                  std::complex<float>* WORK,
                  int LWORK,
                  int& INFO)
  {
    INFO=LAPACKE_cungqr_work(CblasColMajor, M, N, K, MM(A), LDA, MM(TAU), MM(WORK), LWORK);
  }

  static void gqr(int M, int N, int K, float* A, const int LDA, float* TAU, float* WORK, int LWORK, int& INFO)
  {
    INFO=LAPACKE_sorgqr_work(CblasColMajor, M, N, K, A, LDA, TAU, WORK, LWORK);
  }

  static void glq(int M,
                  int N,
                  int K,
                  std::complex<double>* A,
                  const int LDA,
                  std::complex<double>* TAU,
                  std::complex<double>* WORK,
                  int LWORK,
                  int& INFO)
  {
    INFO=LAPACKE_zunglq_work(CblasColMajor, M, N, K, MM(A), LDA, MM(TAU), MM(WORK), LWORK);
  }

  static void glq(int M, int N, int K, double* A, const int LDA, double* TAU, double* WORK, int LWORK, int& INFO)
  {
    INFO=LAPACKE_dorglq_work(CblasColMajor, M, N, K, A, LDA, TAU, WORK, LWORK);
  }

  static void glq(int M,
                  int N,
                  int K,
                  std::complex<float>* A,
                  const int LDA,
                  std::complex<float>* TAU,
                  std::complex<float>* WORK,
                  int LWORK,
                  int& INFO)
  {
    INFO=LAPACKE_cunglq_work(CblasColMajor, M, N, K, MM(A), LDA, MM(TAU), MM(WORK), LWORK);
  }

  static void glq(int M, int N, int K, float* A, const int LDA, float* TAU, float* WORK, int const LWORK, int& INFO)
  {
    INFO=LAPACKE_sorglq_work(CblasColMajor, M, N, K, A, LDA, TAU, WORK, LWORK);
  }

  static void potrf(const char& UPLO, const int& N, float* A, const int& LDA, int& INFO)
  {
    INFO=LAPACKE_spotrf(CblasColMajor,UPLO, N, A, LDA);
  }

  static void potrf(const char& UPLO, const int& N, double* A, const int& LDA, int& INFO)
  {
    INFO=LAPACKE_dpotrf(CblasColMajor,UPLO, N, A, LDA);
  }

  static void potrf(const char& UPLO, const int& N, std::complex<float>* A, const int& LDA, int& INFO)
  {
    INFO=LAPACKE_cpotrf(CblasColMajor,UPLO, N, MM(A), LDA);
  }

  static void potrf(const char& UPLO, const int& N, std::complex<double>* A, const int& LDA, int& INFO)
  {
    INFO=LAPACKE_zpotrf(CblasColMajor,UPLO, N, MM(A), LDA);
  }
};


#endif // OHMMS_BLAS_H
