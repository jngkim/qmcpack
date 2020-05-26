//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//    Lawrence Livermore National Laboratory
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov
//    Lawrence Livermore National Laboratory
////////////////////////////////////////////////////////////////////////////////

#ifndef HIPSPARSE_FUNCTIONDEFS_H
#define HIPSPARSE_FUNCTIONDEFS_H

#include<cassert>
#include <hip/hip_runtime.h>
#include "hipsparse.h"
#include "AFQMC/Memory/HIP/hip_utilities.h"

namespace hipsparse {

  using qmc_hip::hipsparseOperation;

  // Level-2
  inline hipsparseStatus_t hipsparse_csrmv(hipsparseHandle_t handle, char Atrans,
               int m, int n, int nnz, const double alpha,
               const hipsparseMatDescr_t &descrA,
               const double *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const double *x, const double beta, double *y)

  {
    hipsparseStatus_t sucess =
                hipsparseDcsrmv(handle,hipsparseOperation(Atrans),m,n,nnz,&alpha,
                               descrA,csrValA,csrRowPtrA,csrColIndA,x,&beta,y);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_csrmv(hipsparseHandle_t handle, char Atrans,
               int m, int n, int nnz, const float alpha,
               const hipsparseMatDescr_t &descrA,
               const float *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const float *x, const float beta, float *y)

  {
    hipsparseStatus_t sucess =
                hipsparseScsrmv(handle,hipsparseOperation(Atrans),m,n,nnz,&alpha,
                               descrA,csrValA,csrRowPtrA,csrColIndA,x,&beta,y);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_csrmv(hipsparseHandle_t handle, char Atrans,
               int m, int n, int nnz, const std::complex<double> alpha,
               const hipsparseMatDescr_t &descrA,
               const std::complex<double> *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const std::complex<double> *x, const std::complex<double> beta,
               std::complex<double> *y)

  {
    hipsparseStatus_t sucess =
                hipsparseZcsrmv(handle,hipsparseOperation(Atrans),m,n,nnz,
                               reinterpret_cast<hipDoubleComplex const*>(&alpha),
                               descrA,
                               reinterpret_cast<hipDoubleComplex const*>(csrValA),
                               csrRowPtrA,csrColIndA,
                               reinterpret_cast<hipDoubleComplex const*>(x),
                               reinterpret_cast<hipDoubleComplex const*>(&beta),
                               reinterpret_cast<hipDoubleComplex *>(y));
    hipDeviceSynchronize ();
    return sucess;
  }


  inline hipsparseStatus_t hipsparse_csrmv(hipsparseHandle_t handle, char Atrans,
               int m, int n, int nnz, const std::complex<float> alpha,
               const hipsparseMatDescr_t &descrA,
               const std::complex<float> *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const std::complex<float> *x, const std::complex<float> beta,
               std::complex<float> *y)
  {
    hipsparseStatus_t sucess =
                hipsparseCcsrmv(handle,hipsparseOperation(Atrans),m,n,nnz,
                               reinterpret_cast<hipComplex const*>(&alpha),
                               descrA,
                               reinterpret_cast<hipComplex const*>(csrValA),
                               csrRowPtrA,csrColIndA,
                               reinterpret_cast<hipComplex const*>(x),
                               reinterpret_cast<hipComplex const*>(&beta),
                               reinterpret_cast<hipComplex *>(y));
    hipDeviceSynchronize ();
    return sucess;
  }


  inline hipsparseStatus_t hipsparse_csrmm(hipsparseHandle_t handle, char Atrans,
               int m, int n, int k, int nnz, const double alpha,
               const hipsparseMatDescr_t &descrA,
               const double *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const double *B, const int ldb,
               const double beta,
               double *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseDcsrmm(handle,hipsparseOperation(Atrans),
                               m,n,k,nnz,&alpha,descrA,csrValA,csrRowPtrA,csrColIndA,
                               B,ldb,&beta,C,ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_csrmm(hipsparseHandle_t handle, char Atrans,
               int m, int n, int k, int nnz, const float alpha,
               const hipsparseMatDescr_t &descrA,
               const float *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const float *B, const int ldb,
               const float beta,
               float *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseScsrmm(handle,hipsparseOperation(Atrans),
                               m,n,k,nnz,&alpha,descrA,csrValA,csrRowPtrA,csrColIndA,
                               B,ldb,&beta,C,ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_csrmm(hipsparseHandle_t handle, char Atrans,
               int m, int n, int k, int nnz, const std::complex<double> alpha,
               const hipsparseMatDescr_t &descrA,
               const std::complex<double> *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const std::complex<double> *B, const int ldb,
               const std::complex<double> beta,
               std::complex<double> *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseZcsrmm(handle,hipsparseOperation(Atrans),
                               m,n,k,nnz,
                               reinterpret_cast<hipDoubleComplex const*>(&alpha),
                               descrA,
                               reinterpret_cast<hipDoubleComplex const*>(csrValA),
                               csrRowPtrA,csrColIndA,
                               reinterpret_cast<hipDoubleComplex const*>(B),ldb,
                               reinterpret_cast<hipDoubleComplex const*>(&beta),
                               reinterpret_cast<hipDoubleComplex *>(C),ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_csrmm(hipsparseHandle_t handle, char Atrans,
               int m, int n, int k, int nnz, const std::complex<float> alpha,
               const hipsparseMatDescr_t &descrA,
               const std::complex<float> *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const std::complex<float> *B, const int ldb,
               const std::complex<float> beta,
               std::complex<float> *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseCcsrmm(handle,hipsparseOperation(Atrans),
                               m,n,k,nnz,
                               reinterpret_cast<hipComplex const*>(&alpha),
                               descrA,
                               reinterpret_cast<hipComplex const*>(csrValA),
                               csrRowPtrA,csrColIndA,
                               reinterpret_cast<hipComplex const*>(B),ldb,
                               reinterpret_cast<hipComplex const*>(&beta),
                               reinterpret_cast<hipComplex *>(C),ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_csrmm2(hipsparseHandle_t handle, char Atrans,
               char Btrans, int m, int n, int k, int nnz, const double alpha,
               const hipsparseMatDescr_t &descrA,
               const double *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const double *B, const int ldb,
               const double beta,
               double *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseDcsrmm2(handle,hipsparseOperation(Atrans),hipsparseOperation(Btrans),
                               m,n,k,nnz,&alpha,descrA,csrValA,csrRowPtrA,csrColIndA,
                               B,ldb,&beta,C,ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_csrmm2(hipsparseHandle_t handle, char Atrans,
               char Btrans, int m, int n, int k, int nnz, const float alpha,
               const hipsparseMatDescr_t &descrA,
               const float *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const float *B, const int ldb,
               const float beta,
               float *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseScsrmm2(handle,hipsparseOperation(Atrans),hipsparseOperation(Btrans),
                               m,n,k,nnz,&alpha,descrA,csrValA,csrRowPtrA,csrColIndA,
                               B,ldb,&beta,C,ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_csrmm2(hipsparseHandle_t handle, char Atrans,
               char Btrans, int m, int n, int k, int nnz, const std::complex<double> alpha,
               const hipsparseMatDescr_t &descrA,
               const std::complex<double> *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const std::complex<double> *B, const int ldb,
               const std::complex<double> beta,
               std::complex<double> *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseZcsrmm2(handle,hipsparseOperation(Atrans),hipsparseOperation(Btrans),
                               m,n,k,nnz,
                               reinterpret_cast<hipDoubleComplex const*>(&alpha),
                               descrA,
                               reinterpret_cast<hipDoubleComplex const*>(csrValA),
                               csrRowPtrA,csrColIndA,
                               reinterpret_cast<hipDoubleComplex const*>(B),ldb,
                               reinterpret_cast<hipDoubleComplex const*>(&beta),
                               reinterpret_cast<hipDoubleComplex *>(C),ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_csrmm2(hipsparseHandle_t handle, char Atrans,
               char Btrans, int m, int n, int k, int nnz, const std::complex<float> alpha,
               const hipsparseMatDescr_t &descrA,
               const std::complex<float> *csrValA,
               const int *csrRowPtrA, const int *csrColIndA,
               const std::complex<float> *B, const int ldb,
               const std::complex<float> beta,
               std::complex<float> *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseCcsrmm2(handle,hipsparseOperation(Atrans),hipsparseOperation(Btrans),
                               m,n,k,nnz,
                               reinterpret_cast<hipComplex const*>(&alpha),
                               descrA,
                               reinterpret_cast<hipComplex const*>(csrValA),
                               csrRowPtrA,csrColIndA,
                               reinterpret_cast<hipComplex const*>(B),ldb,
                               reinterpret_cast<hipComplex const*>(&beta),
                               reinterpret_cast<hipComplex *>(C),ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

  // TODO: FDM Not implemnted.
  inline hipsparseStatus_t hipsparse_gemmi(hipsparseHandle_t handle,
               int m, int n, int k, int nnz, const double alpha,
               const double *A, const int lda,
               const double *cscValB,
               const int *cscColPtrB, const int *cscRowIndB,
               const double beta,
               double *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseDgemmi(handle,m,n,k,nnz,&alpha,A,lda,cscValB,
                               cscColPtrB,cscRowIndB,&beta,C,ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_gemmi(hipsparseHandle_t handle,
               int m, int n, int k, int nnz, const float alpha,
               const float *A, const int lda,
               const float *cscValB,
               const int *cscColPtrB, const int *cscRowIndB,
               const float beta,
               float *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseSgemmi(handle,m,n,k,nnz,&alpha,A,lda,cscValB,
                               cscColPtrB,cscRowIndB,&beta,C,ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_gemmi(hipsparseHandle_t handle,
               int m, int n, int k, int nnz, const std::complex<double> alpha,
               const std::complex<double> *A, const int lda,
               const std::complex<double> *cscValB,
               const int *cscColPtrB, const int *cscRowIndB,
               const std::complex<double> beta,
               std::complex<double> *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseZgemmi(handle,m,n,k,nnz,
                               reinterpret_cast<hipDoubleComplex const*>(&alpha),
                               reinterpret_cast<hipDoubleComplex const*>(A),lda,
                               reinterpret_cast<hipDoubleComplex const*>(cscValB),
                               cscColPtrB,cscRowIndB,
                               reinterpret_cast<hipDoubleComplex const*>(&beta),
                               reinterpret_cast<hipDoubleComplex *>(C),ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

  inline hipsparseStatus_t hipsparse_gemmi(hipsparseHandle_t handle,
               int m, int n, int k, int nnz, const std::complex<float> alpha,
               const std::complex<float> *A, const int lda,
               const std::complex<float> *cscValB,
               const int *cscColPtrB, const int *cscRowIndB,
               const std::complex<float> beta,
               std::complex<float> *C, const int ldc)

  {
    hipsparseStatus_t sucess =
                hipsparseCgemmi(handle,m,n,k,nnz,
                               reinterpret_cast<hipComplex const*>(&alpha),
                               reinterpret_cast<hipComplex const*>(A),lda,
                               reinterpret_cast<hipComplex const*>(cscValB),
                               cscColPtrB,cscRowIndB,
                               reinterpret_cast<hipComplex const*>(&beta),
                               reinterpret_cast<hipComplex *>(C),ldc);
    hipDeviceSynchronize ();
    return sucess;
  }

}

#endif
