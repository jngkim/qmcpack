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
#include "OMPTarget/OMPallocator.hpp"
#include "SYCL/SYCLDeviceManager.hpp"
#include "SYCL/syclBLAS.hpp"
#include "SYCL/SYCLallocator.hpp"
#include <OhmmsPETE/OhmmsVector.h>
#include <OhmmsPETE/OhmmsMatrix.h>
#include <CPU/BLAS.hpp>
#include "oneapi/mkl/blas.hpp"
#include "mkl.h"

namespace qmcplusplus
{

template<typename T, typename Alloc>
void test_gemv(const int M_b, const int N_b, const char trans)
{
  const int M = trans == 'T' ? M_b : N_b;
  const int N = trans == 'T' ? N_b : M_b;

  using vec_t = Vector<T, Alloc>;
  using mat_t = Matrix<T, Alloc>;

  sycl::queue *handle=get_default_queue();

  vec_t A(N);        // Input vector
  mat_t B(M_b, N_b); // Input matrix
  vec_t C(M);        // Result vector ompBLAS
  vec_t D(M);        // Result vector BLAS

  // Fill data
  for (int i = 0; i < N; i++)
    A[i] = i;

  for (int j = 0; j < M_b; j++)
    for (int i = 0; i < N_b; i++)
      B[j][i] = i + j * 2;

  // Fill C and D with 0
  for (int i = 0; i < M; i++)
    C[i] = D[i] = T(-0.1);

  A.updateTo();
  B.updateTo();

  T alpha(1);
  T beta(0);

  // in Fortran, B[M][N] is viewed as B^T
  // when trans == 'T', the actual calculation is B * A[N] = C[M]
  // when trans == 'N', the actual calculation is B^T * A[M] = C[N]
  //ompBLAS::gemv(handle, trans, N_b, M_b, alpha, B.device_data(), N_b, A.device_data(), 1, beta, C.device_data(), 1);

  const auto transA = (trans == 'T') ? oneapi::mkl::transpose::trans: oneapi::mkl::transpose::nontrans;
  syclBLAS::gemv(*handle, transA, M_b, M_b, alpha, B.device_data(), N_b, A.device_data(), 1, beta, C.device_data(),1).wait();

  C.updateFrom();

  if (trans == 'T')
    BLAS::gemv_trans(M_b, N_b, B.data(), A.data(), D.data());
  else
    BLAS::gemv(M_b, N_b, B.data(), A.data(), D.data());

  for (int index = 0; index < M; index++)
    CHECK(C[index] == D[index]);
}

template<typename T>
void test_gemv_batched(const int M_b, const int N_b, const char trans, const int batch_count)
{
  const int M = trans == 'T' ? M_b : N_b;
  const int N = trans == 'T' ? N_b : M_b;

  using vec_t = Vector<T, OMPallocator<T>>;
  using mat_t = Matrix<T, OMPallocator<T>>;

  sycl::queue *handle=get_default_queue();

  // Create input vector
  std::vector<vec_t> As;
  Vector<const T*, OMPallocator<const T*>> Aptrs;

  // Create input matrix
  std::vector<mat_t> Bs;
  Vector<const T*, OMPallocator<const T*>> Bptrs;

  // Create output vector (ompBLAS)
  std::vector<vec_t> Cs;
  Vector<T*, OMPallocator<T*>> Cptrs;

  // Create output vector (BLAS)
  std::vector<vec_t> Ds;
  Vector<T*, OMPallocator<T*>> Dptrs;

  // Resize pointer vectors
  Aptrs.resize(batch_count);
  Bptrs.resize(batch_count);
  Cptrs.resize(batch_count);
  Dptrs.resize(batch_count);

  // Resize data vectors
  As.resize(batch_count);
  Bs.resize(batch_count);
  Cs.resize(batch_count);
  Ds.resize(batch_count);

  // Fill data
  for (int batch = 0; batch < batch_count; batch++)
  {

    As[batch].resize(N);
    Aptrs[batch] = As[batch].device_data();

    Bs[batch].resize(M_b, N_b);
    Bptrs[batch] = Bs[batch].device_data();

    Cs[batch].resize(M);
    Cptrs[batch] = Cs[batch].device_data();

    Ds[batch].resize(M);
    Dptrs[batch] = Ds[batch].data();

    for (int i = 0; i < N; i++)
      As[batch][i] = i;

    for (int j = 0; j < M_b; j++)
      for (int i = 0; i < N_b; i++)
        Bs[batch][j][i] = i + j * 2;

    for (int i = 0; i < M; i++)
      Cs[batch][i] = Ds[batch][i] = T(0);

    As[batch].updateTo();
    Bs[batch].updateTo();
    //Skil updateTo on C as beta=0
    //Cs[batch].updateTo();
  }

  Aptrs.updateTo();
  Bptrs.updateTo();
  Cptrs.updateTo();

  T alpha=T(1);
  T beta=T(0);

  const auto transA = (trans == 'T') ? oneapi::mkl::transpose::trans: oneapi::mkl::transpose::nontrans;

  syclBLAS::gemv_batched(*handle, transA, N_b, M_b, 
      &alpha, Bptrs.device_data(), N_b, Aptrs.device_data(), 1, &beta, Cptrs.device_data(), 1, batch_count).wait();

  for (int batch = 0; batch < batch_count; batch++)
  {
    Cs[batch].updateFrom();
    if (trans == 'T')
      BLAS::gemv_trans(M_b, N_b, Bs[batch].data(), As[batch].data(), Ds[batch].data());
    else
      BLAS::gemv(M_b, N_b, Bs[batch].data(), As[batch].data(), Ds[batch].data());

    // Check results
    for (int index = 0; index < M; index++)
      CHECK(Cs[batch][index] == Ds[batch][index]);
  }
}

TEST_CASE("OmpSYCL gemv", "[SYCL]")
{
  const int M           = 137;
  const int N           = 79;
  const int batch_count = 23;

  // Non-batched test
  std::cout << "Testing TRANS gemv" << std::endl;
  test_gemv<float,OMPallocator<float>>(M, N, 'T');
  test_gemv<double,OMPallocator<double>>(M, N, 'T');
#if defined(QMC_COMPLEX)
  test_gemv<std::complex<float>, OMPallocator<std::complex<float>>>(N, M, 'T');
  test_gemv<std::complex<double>,OMPallocator<std::complex<double>>>(N, M, 'T');
#endif

  // batched
  std::cout << "Testing TRANS gemv_batched" << std::endl;
  test_gemv_batched<float>(M, N, 'T', batch_count);
  test_gemv_batched<double>(M, N, 'T', batch_count);
#if defined(QMC_COMPLEX)
  test_gemv_batched<std::complex<float>>(M, N, 'T', batch_count);
  test_gemv_batched<std::complex<double>>(M, N, 'T', batch_count);
#endif
}
} // namespace qmcplusplus
