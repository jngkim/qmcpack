//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Thomas Applencourt, tappencourt@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp
////////////////////////////////////////////////////////////////////////////////////// // -*- C++ -*-
#ifndef QMCPLUSPLUS_SYCL_DEVICE_MANAGER_H
#define QMCPLUSPLUS_SYCL_DEVICE_MANAGER_H
#include <vector>
#include <CL/sycl.hpp>
namespace qmcplusplus
{
  sycl::queue* get_default_queue();

  struct syclDeviceInfo {
    sycl::context sycl_context;
    sycl::device sycl_device;
  };

  std::vector<struct syclDeviceInfo> xomp_get_sycl_devices();

  inline syclDeviceInfo xomp_get_device_info(const int n) {
    return xomp_get_sycl_devices()[n];
  }

}
#endif
