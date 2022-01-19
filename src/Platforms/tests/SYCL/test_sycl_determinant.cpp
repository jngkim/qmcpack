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
#include "QMCWaveFunctions/detail/SYCL/sycl_determinant_helper.hpp"


namespace qmcplusplus
{

  template<typename T>
    void test_detlog(int M)
    {
      Vector<T,OMPallocator<T>>  A(M*M);
      Vector<std::int64_t,OMPallocator<std::int64_t>>  Pivots(M);
      std::iota(Pivots.begin(),Pivots.end(),1);

      std::swap(Pivots[2],Pivots[3]);

      {
        std::mt19937 rng;
        std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
        std::generate_n(A.data(),A.size(),[&]() { return udist(rng);});
      }

      auto res=computeLogDet<T>(M,M,A.data(),Pivots.data());

      A.updateTo();
      Pivots.updateTo();

      sycl::queue m_queue{*get_default_queue()};
      auto res_gpu= computeLogDet<T>(m_queue,M,M,A.device_data(), Pivots.device_data());

      CHECK(res.real() == Approx(res_gpu.real()));
      CHECK(res.imag() == Approx(res_gpu.imag()));

      res_gpu= computeLogDetNDR<T>(m_queue,M,M,A.device_data(), Pivots.device_data());
      CHECK(res.real() == Approx(res_gpu.real()));
      CHECK(res.imag() == Approx(res_gpu.imag()));

    }

TEST_CASE("OmpSYCL single DetLog", "[SYCL]")
{
  const int M  = 1024;

  test_detlog<double>(M);
}

  template<typename T>
    void test_detlog_batched(int M, int batch_count)
    {
      Vector<T,OMPallocator<T>>  A(M*M*batch_count);
      Vector<std::int64_t,OMPallocator<std::int64_t>>  Pivots(M*batch_count);

      for(int iw=0; iw<batch_count; ++iw)
      {
        auto* pv=Pivots.data()+iw*M;
        std::iota(pv,pv+M,1);
        std::swap(pv[2],pv[3]);
      }

      {
        std::mt19937 rng;
        std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
        std::generate_n(A.data(),A.size(),[&]() { return udist(rng);});
      }

      A.updateTo();
      Pivots.updateTo();

      sycl::queue m_queue{*get_default_queue()};
      const int strideA=M*M;

      Vector<std::complex<T>,OMPallocator<std::complex<T>>>  logdets(batch_count);
      computeLogDet_batched(m_queue,M,M,A.device_data(), Pivots.device_data(), 
          logdets.device_data(), batch_count).wait();
      logdets.updateFrom();

      for(int iw=0; iw<batch_count; ++iw)
      {
        auto res=computeLogDet<T>(M,M,A.data()+iw*strideA,Pivots.data()+iw*M);
        auto res_gpu=logdets[iw];
        CHECK(res.real() == Approx(res_gpu.real()));
        CHECK(res.imag() == Approx(res_gpu.imag()));
      }
#if 0
      //group reduction is broken
      logdets=T{};
      logdets.updateTo();

      computeLogDetGroup(m_queue,M,M,A.device_data(), Pivots.device_data(), 
          logdets.device_data(), batch_count).wait();
      logdets.updateFrom();

      for(int iw=0; iw<batch_count; ++iw)
      {
        auto res=computeLogDet<T>(M,M,A.data()+iw*strideA,Pivots.data()+iw*M);
        auto res_gpu=logdets[iw];
        CHECK(res.real() == Approx(res_gpu.real()));
        CHECK(res.imag() == Approx(res_gpu.imag()));
      }
#endif
    }

TEST_CASE("OmpSYCL batched DetLog", "[SYCL]")
{
  const int M  = 256;
  const int batch_count = 256;

  test_detlog_batched<double>(M,batch_count);
}

} // namespace qmcplusplus
