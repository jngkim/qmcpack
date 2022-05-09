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
#include "Platforms/CPU/SIMD/aligned_allocator.hpp"
#include "OMPTarget/OMPallocator.hpp"
#include "Platforms/CPU/SIMD/simd.hpp"
#include "QMCWaveFunctions/Fermion/syclSolverInverter.hpp"
#include "checkMatrix.hpp"
#include "createTestMatrix.h"

namespace qmcplusplus
{
using LogValue = std::complex<double>;

#if 0
TEMPLATE_TEST_CASE("syclSolverInverter", "[wavefunction][fermion]", double, float)
{
  // TestType is defined by Catch. It is the type in each instantiation of the templated test case.
#ifdef QMC_COMPLEX
  using FullPrecValue = std::complex<double>;
  using Value         = std::complex<TestType>;
#else
  using FullPrecValue = double;
  using Value         = TestType;
#endif
  sycl::queue m_queue = getSYCLDefaultDeviceDefaultQueue();
  syclSolverInverter<FullPrecValue> solver;
  const int N = 3;

  Matrix<Value> m(N, N);
  Matrix<Value> m_invT(N, N);
  Matrix<Value, SYCLAllocator<Value>> m_invGPU(N, N);
  LogValue log_value;

  SECTION("3x3 matrix")
  {
    Matrix<Value> a(N, N);
    Matrix<Value> a_T(N, N);
    TestMatrix1::fillInput(a);

    simd::transpose(a.data(), a.rows(), a.cols(), a_T.data(), a_T.rows(), a_T.cols());
    solver.invert_transpose(a_T, m_invT, m_invGPU, log_value, m_queue);
    REQUIRE(log_value == LogComplexApprox(TestMatrix1::logDet()));

    Matrix<Value> b(3, 3);

    TestMatrix1::fillInverse(b);

    auto check_matrix_result = checkMatrix(m_invT, b);
    CHECKED_ELSE(check_matrix_result.result) { FAIL(check_matrix_result.result_message); }
  }
}
#endif

#if 1
template<typename T, typename T_FP>
void test_inverse(const std::int64_t M)
{
  sycl::queue m_queue = getSYCLDefaultDeviceDefaultQueue();

  Matrix<T> A(M,M); 
  Matrix<T> B(M,M); //for validation

  { 
    std::mt19937 rng;
    std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
    std::generate_n(B.data(),B.size(),[&]() { return udist(rng);});
    std::copy_n(B.data(), B.size(), A.data());
  }

  mklSolverInverter<T_FP> diag_eng;

  Matrix<T> Ainv;
  Matrix<T,SYCLAllocator<T>> Ainv_gpu;
  Ainv.resize(M,M);

  std::complex<double> log_value;

  diag_eng.invert_transpose(A, Ainv, Ainv_gpu, log_value, m_queue);
  m_queue.wait();
  m_queue.memcpy(A.data(),Ainv.device_data(),Ainv.size()*sizeof(T)).wait();

  //check the identity
  Matrix<T> C(M,M);
  BLAS::gemm('T', 'N', M, M, M, 1.0, B.data(), M, A.data(), M, 0.0, C.data(),M);

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

#endif

} // namespace qmcplusplus
