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
#include "SYCL/SYCLruntime.hpp"
#include <OhmmsPETE/OhmmsVector.h>
#include <OhmmsPETE/OhmmsMatrix.h>
#include <CPU/BLAS.hpp>
#include "oneapi/mkl/rng.hpp"
#include "oneapi/mkl/lapack.hpp"
#include "mkl.h"


namespace qmcplusplus
{

namespace syclSolver=oneapi::mkl::lapack;

template<typename T, typename Alloc>
void test_inverse(const std::int64_t M)
{
  using mat_t = Matrix<T, Alloc>;
  using vec_t = Vector<T, Alloc>;

  sycl::queue *handle=get_default_queue();

  mat_t A(M,M);
  mat_t B(M,M);

  oneapi::mkl::rng::mt19937 rng{*handle};
  oneapi::mkl::rng::uniform<T> udist{T(-0.5),T(0.5)}; 

  oneapi::mkl::rng::generate(udist, rng, A.size(), A.device_data()).wait(); 
  A.updateFrom();
  std::copy(A.data(),A.data()+A.size(), B.data()); //save B

  auto getrf_ws=syclSolver::getrf_scratchpad_size<T>(*handle,M,M,M);
  auto getri_ws=syclSolver::getri_scratchpad_size<T>(*handle,M,M);

  Vector<std::int64_t,OMPallocator<std::int64_t>> pivots(M);
  vec_t workspace(std::max(getrf_ws,getri_ws));

  auto e = syclSolver::getrf(*handle,M,M,A.device_data(),M,pivots.device_data(), workspace.device_data(), getrf_ws);
  syclSolver::getri(*handle,M,A.device_data(),M,pivots.device_data(), workspace.device_data(), getri_ws, {e}).wait();
  A.updateFrom();

  //check the identity
  mat_t C(M,M);
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

TEST_CASE("OmpSYCL solver", "[SYCL]")
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
} // namespace qmcplusplus
