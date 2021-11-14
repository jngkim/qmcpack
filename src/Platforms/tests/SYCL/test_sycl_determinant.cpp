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
#include "QMCWaveFunctions/detail/SYCL/determinant_helper.hpp"


namespace qmcplusplus
{

TEST_CASE("OmpSYCL determinant values", "[SYCL]")
{
  const int M           = 1024;

  using T=double;

  Vector<T,OMPallocator<T>>  A(M*M);
  Vector<std::int64_t,OMPallocator<std::int64_t>>  Pivots(M);
  std::iota(Pivots.begin(),Pivots.end(),1);

  std::swap(Pivots[2],Pivots[3]);

  {
    std::mt19937 rng;
    std::uniform_real_distribution<T> udist{T(-0.5),T(0.5)}; 
    std::generate_n(A.data(),A.size(),[&]() { return udist(rng);});
  }

  auto res=computeLogDet<T>(A.data(),M,M,Pivots.data());

  A.updateTo();
  Pivots.updateTo();

  sycl::queue m_queue{*get_default_queue()};
  auto res_gpu= computeLogDet<T>(m_queue,A.device_data(), M,M,Pivots.device_data());

  CHECK(res.real() == Approx(res_gpu.real()));
  CHECK(res.imag() == Approx(res_gpu.imag()));

  res_gpu= computeLogDet_ND<T>(m_queue,A.device_data(), M,M,Pivots.device_data());

  CHECK(res.real() == Approx(res_gpu.real()));
  CHECK(res.imag() == Approx(res_gpu.imag()));
}

} // namespace qmcplusplus
