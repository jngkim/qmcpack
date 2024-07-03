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
#include <Containers/OhmmsPETE/OhmmsVector.h>
#include <Containers/OhmmsPETE/OhmmsMatrix.h>
#include "DeviceManager.h"
#include "SYCL/syclBLAS.hpp"
#include "CPU/BLAS.hpp"
#include "QMCWaveFunctions/Fermion/syclSolverInverter.hpp"
#include "QMCWaveFunctions/Fermion/syclSolverInverterRS.hpp"

namespace qmcplusplus
{

template<typename T, typename DigEng>
void test_inverse(const std::int64_t M)
{
  sycl::queue m_queue{DeviceManager::getGlobal().getSYCLDM().createQueueDefaultDevice()};

  Matrix<T> A(M,M); 
  Matrix<T> B(M,M); //for validation

  { 
    std::mt19937 rng;
    std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
    std::generate_n(B.data(),B.size(),[&]() { return udist(rng);});
    std::copy_n(B.data(), B.size(), A.data());
  }

  DigEng diag_eng;

  Matrix<T> Ainv;
  Matrix<T,SYCLAllocator<T>> Ainv_gpu;
  Ainv.resize(M,M);
  Ainv_gpu.resize(M,M);

  std::complex<double> log_value;

  diag_eng.invert_transpose(A, Ainv, Ainv_gpu, log_value, m_queue);
  m_queue.wait();

  //check the identity
  Matrix<T> C(M,M);
  BLAS::gemm('T', 'N', M, M, M, 1.0, B.data(), M, Ainv.data(), M, 0.0, C.data(),M);

  for(int i=0; i<M; ++i)
  {
    for(int j=0; j<M; ++j)
      if(i==j) 
        CHECK(C[i][j] == Approx(1.0));
      else
        CHECK(C[i][j] == Approx(0.0));
  }

}

TEST_CASE("OmpSYCL mklSolverInverter", "[SYCL]")
{
  const int M           = 911;

#ifndef ONEMKL_CUDA_MISSING
  test_inverse<float,syclSolverInverter<double>>(M);
  test_inverse<double,syclSolverInverter<double>>(M);
#endif
  test_inverse<float,syclSolverInverterRS<double>>(M);
  test_inverse<double,syclSolverInverterRS<double>>(M);
}


} // namespace qmcplusplus
