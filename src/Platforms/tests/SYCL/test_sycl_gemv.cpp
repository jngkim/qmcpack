//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#include "catch.hpp"

#include <memory>
#include <vector>
#include <iostream>
#include "DeviceManager.h"
#include "SYCL/SYCLruntime.hpp"
#include "SYCL/SYCLallocator.hpp"
#include "SYCL/syclBLAS.hpp"
#include "oneapi/mkl/blas.hpp"
#include "CPU/BLAS.hpp"
#include "OhmmsPETE/OhmmsVector.h"
#include "OhmmsPETE/OhmmsMatrix.h"


namespace qmcplusplus
{

//Moved to SYCL/syclBLAS.hpp/cpp
//namespace syclBLAS
//{
//template<typename T>
//inline sycl::event gemv_batched(sycl::queue&   handle,
//                                const char          trans,
//                                const syclBLAS_int  m,
//                                const syclBLAS_int  n,
//                                const T*            alpha,
//                                const T**           A,
//                                const syclBLAS_int  lda,
//                                const T**           X,
//                                const syclBLAS_int  incx,
//                                const T*            beta,
//                                T**                 Y,
//                                const syclBLAS_int  incy,
//                                const syclBLAS_int  batch_count,
//                                const std::vector<sycl::event> &events = {})
//{
//  oneapi::mkl::transpose trA = trans == 'T' ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
//  //using group api: only one group
//  return oneapi::mkl::blas::gemv_batch(handle, &trA, &m, &n, alpha, A, &lda,
//                                       X,&incx, beta, Y, &incy,1,&batch_count, events);
//}
//}


template<typename T, typename Alloc>
void test_gemv(const int M_b, const int N_b, const char trans)
{
  const int M = trans == 'T' ? M_b : N_b;
  const int N = trans == 'T' ? N_b : M_b;

  using vec_t = Vector<T, Alloc>;
  using mat_t = Matrix<T, Alloc>;

  sycl::queue m_queue{DeviceManager::getGlobal().getSYCLDM().createQueueDefaultDevice()};

  mat_t A(M_b, N_b); // Input matrix
  vec_t X(N);        // Input vector
  vec_t Y(M);        // Result vector ompBLAS


  Matrix<T> A_h(M_b, N_b); // Input matrix
  Vector<T> X_h(N);        // Input vector
  Vector<T> Y_h(M);        // Result vector ompBLAS

  Vector<T> Y_c(M);        // Result vector ompBLAS

  for (int i = 0; i < N; i++)
    X_h[i] = i;

  for (int j = 0; j < M_b; j++)
    for (int i = 0; i < N_b; i++)
      A_h[j][i] = (i + j*2);

  // Fill C and D with 0
  for (int i = 0; i < M; i++)
  {
    Y_h[i] = T(-0.1);
    Y_c[i] = 0;
  }

  m_queue.memcpy(A.data(), A_h.data(), A.size()*sizeof(T)).wait();
  m_queue.memcpy(X.data(), X_h.data(), X.size()*sizeof(T)).wait();
  m_queue.memcpy(Y.data(), Y_h.data(), Y.size()*sizeof(T)).wait();

  T alpha(1);
  T beta(-1);

  // in Fortran, B[M][N] is viewed as B^T
  // when trans == 'T', the actual calculation is B * A[N] = C[M]
  // when trans == 'N', the actual calculation is B^T * A[M] = C[N]
  //ompBLAS::gemv(handle, trans, N_b, M_b, alpha, B.device_data(), N_b, A.device_data(), 1, beta, C.device_data(), 1);
  syclBLAS::gemv(m_queue, trans, N_b, M_b, alpha, A.data(), N_b, X.data(), 1, beta, Y.data(),1).wait();
  m_queue.memcpy(Y_c.data(), Y.data(), Y.size()*sizeof(T)).wait();

  BLAS::gemv(trans, N_b, M_b, alpha, A_h.data(),  N_b,  X_h.data(), 1, beta, Y_h.data(),1);

  for (int index = 0; index < M; index++)
    CHECK(Y_c[index] == Approx(Y_h[index]));
}

TEST_CASE("OmpSYCL gemv", "[SYCL]")
{
  const int M           = 911;
  const int N           = 64;
  const int batch_count = 8;

  // Non-batched test
  std::cout << "Testing TRANS gemv" << std::endl;
  test_gemv<float,SYCLAllocator<float>>(M, N, 'T');
  test_gemv<double,SYCLAllocator<double>>(M, N, 'T');
  test_gemv<float,SYCLAllocator<float>>(N, N, 'N');
  test_gemv<float,SYCLAllocator<float>>(M, N, 'N');
}

template<typename T, typename Alloc>
void test_gemv_batched(int M_b, int N_b, int batch_size, const char trans)
{
  const int M = trans == 'T' ? M_b : N_b;
  const int N = trans == 'T' ? N_b : M_b;

  using vec_t = Vector<T, Alloc>;
  using mat_t = Matrix<T, Alloc>;

  sycl::queue m_queue{DeviceManager::getGlobal().getSYCLDM().createQueueDefaultDevice()};

  Matrix<T> A_h(M_b, N_b); // Input matrix
  Vector<T> X_h(N);        // Input vector
  Vector<T> Y_h(M);        // Result vector ompBLAS
  Vector<T> Y_c(M);        // Result vector ompBLAS

  T norm = 1.0/(M_b+N_b);
  for (int i = 0; i < N; i++)
    X_h[i] = i*norm;

  for (int j = 0; j < M_b; j++)
    for (int i = 0; i < N_b; i++)
      A_h[j][i] = (i + j*2)*norm;

  // Fill C and D with 0
  for (int i = 0; i < M; i++)
  {
    Y_h[i] = T(-0.1);
    Y_c[i] = 0;
  }

  std::vector<mat_t*> As(batch_size);
  std::vector<vec_t*> Xs(batch_size);
  std::vector<vec_t*> Ys(batch_size);

  Vector<T*,SYCLHostAllocator<T*>> Aptrs(batch_size), Xptrs(batch_size), Yptrs(batch_size);

  for(int batch=0; batch<batch_size; ++batch)
  {
    As[batch]=new mat_t(M_b, N_b);
    Xs[batch]=new vec_t(N);
    Ys[batch]=new vec_t(M);

    m_queue.memcpy(As[batch]->data(), A_h.data(), A_h.size()*sizeof(T)).wait();
    m_queue.memcpy(Xs[batch]->data(), X_h.data(), X_h.size()*sizeof(T)).wait();
    m_queue.memcpy(Ys[batch]->data(), Y_h.data(), Y_h.size()*sizeof(T)).wait();

    Aptrs[batch]=As[batch]->data();
    Xptrs[batch]=Xs[batch]->data();
    Yptrs[batch]=Ys[batch]->data();
  }

  T alpha(1);
  T beta(-1);


  syclBLAS::gemv_batched(m_queue, trans, N_b, M_b, &alpha, (const T**)Aptrs.data(), N_b, (const T**)Xptrs.data(), 1, &beta, Yptrs.data(), 1, batch_size).wait();
  //for(int i=0; i<batch_size; ++i)
  //{
  //  syclBLAS::gemv(*m_queue, trans, N_b, M_b, alpha, As[i]->data(), N_b, Xs[i]->data(), 1, beta, Ys[i]->data(),1).wait();
  //}

  // in Fortran, B[M][N] is viewed as B^T
  // when trans == 'T', the actual calculation is B * A[N] = C[M]
  // when trans == 'N', the actual calculation is B^T * A[M] = C[N]
  //ompBLAS::gemv(handle, trans, N_b, M_b, alpha, B.device_data(), N_b, A.device_data(), 1, beta, C.device_data(), 1);

  BLAS::gemv(trans, N_b, M_b, alpha, A_h.data(),  N_b,  X_h.data(), 1, beta, Y_h.data(),1);

  for(int batch=0; batch<batch_size; ++batch)
  {
    m_queue.memcpy(Y_c.data(), Ys[batch]->data(), Y_c.size()*sizeof(T)).wait();
    for (int index = 0; index < M; index++)
      CHECK(Y_c[index] == Approx(Y_h[index]));
  }

  for(int batch=0; batch<batch_size; ++batch)
  {
    delete As[batch];
    delete Xs[batch];
    delete Ys[batch];
  }
}

TEST_CASE("OmpSYCL gemv_batched", "[SYCL]")
{
  const int M           = 911;
  const int N           = 64;
  const int batch_count = 8;

  // Non-batched test
  std::cout << "Testing TRANS gemv_batched" << std::endl;
  test_gemv_batched<float,SYCLAllocator<float>>(M, N, batch_count, 'T');
  test_gemv_batched<float,SYCLAllocator<float>>(M, N, batch_count, 'N');
  test_gemv_batched<float,SYCLAllocator<float>>(N, N, batch_count, 'N');
}


} // namespace qmcplusplus
