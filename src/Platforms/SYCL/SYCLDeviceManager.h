//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Thomas Applencourt, apl@anl.gov, Argonne National Laboratory
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Thomas Applencourt, apl@anl.gov, Argonne National Laboratory
//
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_SYCLDEVICEMANAGER_H
#define QMCPLUSPLUS_SYCLDEVICEMANAGER_H

#include "config.h"
#include <vector>
#include <memory>
#include <CL/sycl.hpp>
#ifdef ENABLE_OFFLOAD
#include <omp.h>
#endif

//#define ENABLE_INTEROP_FOR_IMMEDIATE_COMMAND_LIST

namespace qmcplusplus
{

#ifdef ENABLE_OFFLOAD
struct syclDeviceInfo
{
  sycl::queue* queue_ = nullptr;
  omp_interop_t interop_;
};
#else
struct syclDeviceInfo
{
  std::unique_ptr<sycl::queue> queue_ ;
};
#endif

/** SYCL device manager
 */
class SYCLDeviceManager
{
  static int sycl_default_device_num;
  static std::vector<syclDeviceInfo> visible_devices;
public:

  SYCLDeviceManager(int& default_device_num, int& num_devices, int local_rank, int local_size);

  static sycl::queue* getDefaultDeviceQueuePtr();

#ifdef ENABLE_OFFLOAD
  //return a queue per thread
  inline static sycl::queue* getDeviceQueue(int ip)
  {
    return visible_devices[ip%visible_devices.size()].queue_;
  }
#endif
};

} // namespace qmcplusplus

#endif
