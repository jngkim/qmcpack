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

#include "Configuration.h"
#include "CPU/SIMD/aligned_allocator.hpp"
#include "QMCWaveFunctions/Fermion/syclSolverInverter.hpp"
#include "QMCWaveFunctions/Fermion/DiracMatrix.h"
#include "Utilities/for_testing/checkMatrix.hpp"
#include "Containers/tests/makeRngSpdMatrix.hpp"

namespace qmcplusplus
{

template<typename T, typename T_FP>
void test_inverse(const std::int64_t N)
{
  sycl::queue m_queue = getSYCLDefaultDeviceDefaultQueue();

  syclSolverInverter<T_FP> solver;

  Matrix<T> m(N, N);
  Matrix<T> m_invT(N, N);
  Matrix<T> m_invT_CPU(N, N);
  Matrix<T, SYCLAllocator<T>> m_invGPU;
  m_invGPU.resize(N,N);

  std::complex<double> log_value, log_value_cpu;
  testing::MakeRngSpdMatrix<T> makeRngSpdMatrix{};
  makeRngSpdMatrix(m);

  solver.invert_transpose(m, m_invT, m_invGPU, log_value, m_queue);
  m_queue.wait();

  DiracMatrix<T_FP> dmat;
  dmat.invert_transpose(m, m_invT_CPU, log_value_cpu);

  REQUIRE(log_value == ComplexApprox(log_value_cpu));

  auto check_matrix_result = checkMatrix(m_invT, m_invT_CPU);
  CHECKED_ELSE(check_matrix_result.result) { FAIL(check_matrix_result.result_message); }
}

TEST_CASE("OmpSYCL syclSolverInverter", "[SYCL]")
{
  const int N           = 911;

#ifdef MIXED_PRECISION
  std::cout << "Testing Inverse for mixed precision " << std::endl;
  test_inverse<float,double>(N);
#else
  std::cout << "Testing Inverse for double double " << std::endl;
  test_inverse<double,double>(N);
#endif
}

} // namespace qmcplusplus
