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
#include <random>
#include "OMPTarget/OMPallocator.hpp"
#include "SYCL/SYCLruntime.hpp"
#include "SYCL/SYCLallocator.hpp"
#include <OhmmsPETE/OhmmsVector.h>
#include <OhmmsPETE/OhmmsMatrix.h>
#include <CPU/BLAS.hpp>
#include "oneapi/mkl/lapack.hpp"
#include "mkl.h"


namespace qmcplusplus
{

namespace syclSolver=oneapi::mkl::lapack;


template<typename T, typename Alloc>
void test_inverse(const std::int64_t M)
{
  sycl::queue handle{*get_default_queue()};

  Matrix<T,Alloc> A(M,M);
  Matrix<T>       B(M,M);

  { //intialize B on host with random numbers
    std::mt19937 rng;
    std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
    std::generate_n(B.data(),B.size(),[&]() { return udist(rng);});
    handle.memcpy(A.device_data(),B.data(), B.size()*sizeof(T)).wait();
  }

  //allocate pivots and workspaces on the device
  auto getrf_ws=syclSolver::getrf_scratchpad_size<T>(handle,M,M,M);
  auto getri_ws=syclSolver::getri_scratchpad_size<T>(handle,M,M);
  Vector<std::int64_t,SYCLAllocator<std::int64_t>> pivots(M);
  Vector<T,SYCLAllocator<T>> workspace(std::max(getrf_ws,getri_ws));

  //getrf (LU) -> getri (inverse)
  auto e = syclSolver::getrf(handle,M,M,A.device_data(),M, pivots.data(), workspace.data(), getrf_ws);
  syclSolver::getri(handle,M,A.device_data(),M,pivots.data(), workspace.data(), getri_ws, {e}).wait();

  //update A on host for comparison
  A.updateFrom();

  //check the identity
  Matrix<T> C(M,M);
  BLAS::gemm('N', 'N', M, M, M, 1.0, B.data(), M, A.data(), M, 0.0, C.data(),M);
  for(int i=0; i<M; ++i)
  {
    for(int j=0; j<M; ++j)
      if(i==j) 
        CHECK(C[i][j] == Approx(1.0));
      else
        CHECK(C[i][j] == Approx(0.0));
  }

}

TEST_CASE("OmpSYCL inverse", "[SYCL]")
{
  const int M           = 16;
  const int batch_count = 23;

  // Non-batched test
  std::cout << "Testing Inversei " << std::endl;
  test_inverse<double,OMPallocator<double>>(M);

#if defined(QMC_COMPLEX)
  test_inverse<std::complex<double>,OMPallocator<std::complex<double>>>(N, M, 'T');
#endif

}

template<typename T, typename Alloc>
void test_inverse_batched(const std::int64_t M, int batch_count)
{
  using mat_t = Matrix<T, Alloc>;
  using vec_t = Vector<T, Alloc>;
  using pvec_t = Vector<std::int64_t, SYCLAllocator<std::int64_t>>;
  using dvec_t = Vector<T, SYCLAllocator<T>>;

  sycl::queue handle{*get_default_queue()};

  std::vector<mat_t> As;
  As.resize(batch_count);

  Matrix<T> B(M,M);
  {
    std::mt19937 rng;
    std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
    std::generate_n(B.data(),B.size(),[&]() { return udist(rng);});
  }

  //pivot vectors
  std::vector<pvec_t> Ps;
  Ps.resize(batch_count);

  //workspace vectors
  std::vector<dvec_t> Ws;
  Ws.resize(batch_count);

  auto getrf_ws=syclSolver::getrf_scratchpad_size<T>(handle,M,M,M);
  auto getri_ws=syclSolver::getri_scratchpad_size<T>(handle,M,M);
  auto ws=std::max(getrf_ws,getri_ws); 

  for (int batch = 0; batch < batch_count; batch++)
  {
    As[batch].resize(M,M);
    handle.memcpy(As[batch].device_data(),B.data(), B.size()*sizeof(T)).wait();
    Ps[batch].resize(M);
    Ws[batch].resize(ws);
  }

  std::vector<sycl::event> lu_events(batch_count);
  for (int batch = 0; batch < batch_count; batch++)
  {
    lu_events[batch] = syclSolver::getrf(handle,M,M,As[batch].device_data(),M,
        Ps[batch].data(), Ws[batch].data(), getrf_ws);
  }
  for (int batch = 0; batch < batch_count; batch++)
  {
    lu_events[batch] = syclSolver::getri(handle,M,As[batch].device_data(),M,
        Ps[batch].data(), Ws[batch].data(), getri_ws,{lu_events[batch]});
  }

  //check the identity
  mat_t C(M,M);

  for (int batch = 0; batch < batch_count; batch++)
  {
    lu_events[batch].wait();
    As[batch].updateFrom();

    BLAS::gemm('N', 'N', M, M, M, 1.0, B.data(), M, As[batch].data(), M, 0.0, C.data(),M);
    for(int i=0; i<M; ++i)
    {
      for(int j=0; j<M; ++j)
        if(i==j) 
          CHECK(C[i][j] == Approx(1.0));
        else
          CHECK(C[i][j] == Approx(0.0));
    }
  }

}

TEST_CASE("OmpSYCL batch-inverse", "[SYCL]")
{
  const int M           = 64;
  const int batch_count = 8;

  // Non-batched test
  std::cout << "Testing Inverse batched " << std::endl;
  test_inverse_batched<double,OMPallocator<double>>(M,batch_count);

#if defined(QMC_COMPLEX)
  test_inverse<std::complex<double>,OMPallocator<std::complex<double>>>(N, M, 'T');
#endif

}

} // namespace qmcplusplus
