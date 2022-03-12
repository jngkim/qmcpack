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
#include "SYCL/SYCLruntime.hpp"
#include "SYCL/SYCLallocator.hpp"
#include "SYCL/syclBLAS.hpp"
#include "CPU/BLAS.hpp"
#include "oneapi/mkl/lapack.hpp"


namespace qmcplusplus
{

namespace syclSolver=oneapi::mkl::lapack;


template<typename T, typename Alloc>
void test_inverse(const std::int64_t M, char trans)
{
  sycl::queue handle{*get_default_queue()};

  Matrix<T,Alloc> A(M,M);
  Matrix<T,Alloc> B(M,M);

  { //intialize B on host with random numbers
    std::mt19937 rng;
    std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
    std::generate_n(B.data(),B.size(),[&]() { return udist(rng);});

    if(trans == 'T')
    {
      B.updateTo();
      syclBLAS::transpose(handle,B.device_data(),M,M,A.device_data(),M,M).wait();
    }
    else
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
  BLAS::gemm(trans, 'N', M, M, M, 1.0, B.data(), M, A.data(), M, 0.0, C.data(),M);
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
  std::cout << "Testing Inverse " << std::endl;
  test_inverse<double,OMPallocator<double>>(M,'N');

  std::cout << "Testing transpoe + Inverse " << std::endl;
  test_inverse<double,OMPallocator<double>>(M,'T');

#if defined(QMC_COMPLEX)
  test_inverse<std::complex<double>,OMPallocator<std::complex<double>>>(N, M, 'T');
#endif

}

template<typename T, typename Alloc>
void test_inverse_batched(const std::int64_t M, int batch_count, char trans)
{
  using mat_t = Matrix<T, Alloc>;
  using vec_t = Vector<T, Alloc>;
  using pvec_t = Vector<std::int64_t, SYCLAllocator<std::int64_t>>;
  using dvec_t = Vector<T, SYCLAllocator<T>>;

  sycl::queue handle{*get_default_queue()};

  std::vector<mat_t> As;
  As.resize(batch_count);

  mat_t B(M,M);
  {
    std::mt19937 rng;
    std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
    std::generate_n(B.data(),B.size(),[&]() { return udist(rng);});
    B.updateTo();
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

  std::vector<sycl::event> lu_events(batch_count);

  for (int batch = 0; batch < batch_count; batch++)
  {
    As[batch].resize(M,M);
    if(trans == 'T')
      lu_events[batch]=syclBLAS::transpose(handle,B.device_data(),M,M,As[batch].device_data(),M,M);
    else
      lu_events[batch]=handle.memcpy(As[batch].device_data(),B.data(), B.size()*sizeof(T));
    Ps[batch].resize(M);
    Ws[batch].resize(ws);
  }

  for (int batch = 0; batch < batch_count; batch++)
  {
    lu_events[batch] = syclSolver::getrf(handle,M,M,As[batch].device_data(),M,
        Ps[batch].data(), Ws[batch].data(), getrf_ws, {lu_events[batch]});
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

    BLAS::gemm(trans, 'N', M, M, M, 1.0, B.data(), M, As[batch].data(), M, 0.0, C.data(),M);
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

TEST_CASE("OmpSYCL inverse-batch", "[SYCL]")
{
  const int M           = 64;
  const int batch_count = 8;

  // Non-batched test
  std::cout << "Testing batched Inverse " << batch_count << std::endl;
  test_inverse_batched<double,OMPallocator<double>>(M,batch_count,'N');
  std::cout << "Testing batched Transpose + Inverse " << batch_count << std::endl;
  test_inverse_batched<double,OMPallocator<double>>(M,batch_count,'T');

#if defined(QMC_COMPLEX)
  test_inverse<std::complex<double>,OMPallocator<std::complex<double>>>(N, M, 'T');
#endif

}

#if 0 
//GRF error
template<typename T, typename Alloc>
void test_inverse_batched_strided(const std::int64_t M, int batch_count)
{
  using mat_t = Matrix<T, Alloc>;
  using vec_t = Vector<T, Alloc>;
  using pvec_t = Vector<std::int64_t, SYCLAllocator<std::int64_t>>;
  using dvec_t = Vector<T, SYCLAllocator<T>>;

  sycl::queue handle{*get_default_queue()};

  Matrix<T> B(M,M);
  {
    std::mt19937 rng;
    std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
    std::generate_n(B.data(),B.size(),[&]() { return udist(rng);});
  }

  const size_t lda=B.cols(); 
  const size_t strideA=B.size();

  dvec_t A(strideA*batch_count);
  pvec_t Pivots(batch_count*strideA);
  dvec_t WorkSpace;

  //int64_t getrf_batch_scratchpad_size(cl::sycl::queue &queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a,
  //                                  int64_t stride_ipiv, int64_t batch_size);
  auto getrf_ws=syclSolver::getrf_batch_scratchpad_size<T>(handle,M,M,lda,strideA,lda,batch_count);

  //int64_t getri_batch_scratchpad_size(cl::sycl::queue &queue, int64_t n, int64_t lda, int64_t stride_a,
  //                                  int64_t stride_ipiv, int64_t ldainv, int64_t stride_ainv, int64_t batch_size);
  auto getri_ws=syclSolver::getri_batch_scratchpad_size<T>(handle,M,lda,strideA,lda,lda,strideA,batch_count);

  auto ws=std::max(getrf_ws,getri_ws); 
  WorkSpace.resize(ws);

  for (int batch = 0; batch < batch_count; batch++)
  {
    handle.memcpy(A.data()+batch*strideA,B.data(),B.size()*sizeof(T)).wait();
  }

  auto e = syclSolver::getrf_batch(handle,M,M,A.data(),lda,strideA,Pivots.data(), strideA, batch_count, WorkSpace.data(),getrf_ws);
  syclSolver::getri_batch(handle, M,A.data(),lda,strideA,Pivots.data(), strideA, batch_count, WorkSpace.data(),getri_ws,{e}).wait();

  //check the identity
  Matrix<T> C(M,M);
  Matrix<T> Ah(M,M);

  for (int batch = 0; batch < batch_count; batch++)
  {
    handle.memcpy(Ah.data(),A.data()+strideA,strideA*sizeof(T)).wait();
    BLAS::gemm('N', 'N', M, M, M, 1.0, B.data(), M, Ah.data(), M, 0.0, C.data(),M);
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

TEST_CASE("OmpSYCL inverse-batch-strided", "[SYCL]")
{
  const int M           = 64;
  const int batch_count = 4;

  // Non-batched test
  std::cout << "Testing Inverse stride/batched " << std::endl;
  test_inverse_batched_strided<double,OMPallocator<double>>(M,batch_count);

#if defined(QMC_COMPLEX)
  test_inverse<std::complex<double>,OMPallocator<std::complex<double>>>(N, M, 'T');
#endif

}
#endif


} // namespace qmcplusplus
