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
#include <OhmmsPETE/OhmmsVector.h>
#include <OhmmsPETE/OhmmsMatrix.h>
#include "SYCL/syclBLAS.hpp"
//#include "QMCWaveFunctions/Fermion/mklSolverInverter.hpp"
#include "mklSolverInverter.hpp"
#include "SYCL/mkl.hpp"

namespace qmcplusplus
{

template<typename T, typename T_FP>
void test_inverse(const std::int64_t M)
{
  sycl::queue* handle=get_default_queue();

  Matrix<T> A(M,M); 
  Matrix<T> B(M,M); //for validation

  { 
    std::mt19937 rng;
    std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
    std::generate_n(B.data(),B.size(),[&]() { return udist(rng);});
    std::copy_n(B.data(), B.size(), A.data());
  }

  mklSolverInverter<T_FP> diag_eng;

  Matrix<T,OMPallocator<T>> Ainv;
  Ainv.resize(M,M);

  std::complex<double> log_value;

  diag_eng.invert_transpose(A, Ainv, log_value);
  handle->wait();
  handle->memcpy(A.data(),Ainv.device_data(),Ainv.size()*sizeof(T)).wait();

  //check the identity
  Matrix<T> C(M,M);
  syclBLAS::gemm('T', 'N', M, M, M, 1.0, B.data(), M, A.data(), M, 0.0, C.data(),M);

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
  const int M           = 16;

  std::cout << "Testing Inverse for miaxed precision " << std::endl;
  test_inverse<float,double>(M);

  std::cout << "Testing Inverse for double double " << std::endl;
  test_inverse<double,double>(M);
}


} // namespace qmcplusplus
