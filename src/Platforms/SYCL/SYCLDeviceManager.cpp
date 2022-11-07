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


#include "SYCLDeviceManager.h"
#include <stdexcept>
#include <string>
#include <algorithm>
#include "config.h"
#include "Platforms/Host/OutputManager.h"
#include "Platforms/Host/determineDefaultDeviceNum.h"

namespace qmcplusplus
{

  //initialize singleton
  int SYCLDeviceManager::sycl_default_device_num = -1;
  std::vector<syclDeviceInfo> SYCLDeviceManager::visible_devices;

  SYCLDeviceManager::SYCLDeviceManager(int& default_device_num, int& num_devices, int local_rank, int local_size)
  {
    if(visible_devices.empty())
    {
#ifdef ENABLE_OFFLOAD
      const int sycl_device_count=omp_get_num_devices();
      sycl_default_device_num = determineDefaultDeviceNum(sycl_device_count, local_rank, local_size);

#ifdef ENABLE_INTEROP_FOR_IMMEDIATE_COMMAND_LIST
      visible_devices.resize(omp_get_max_threads()); 
#else
      visible_devices.resize(sycl_device_count);
#endif
      for(int d=0; d<visible_devices.size(); ++d)
      {
        std::cout << "Creeating interop " << d << std::endl;
        omp_interop_t interop;
#pragma omp interop device(sycl_default_device_num) init(prefer_type("sycl"),targetsync: interop)

        int result;
        sycl::queue* omp_queue = static_cast<sycl::queue *>(omp_get_interop_ptr(interop, omp_ipr_targetsync, &result));
        if(result != omp_irc_success)
          throw std::runtime_error("SYCLDeviceManager::SYCLDeviceManager fail to obtain sycl::queue by interop");

        visible_devices[d].queue_ = omp_queue;
        visible_devices[d].interop_ = interop;
      }
#else
      //simplify it
      visible_devices.resize(1);
      visible_devices[0].queue_.reset(new sycl::queue{sycl::gpu_selector()});
      sycl_default_device_num = 0;
#endif
    }
    else
    {
      std::cout << "SYCLDeviceManager::sycl_default_device_num=" << sycl_default_device_num  << " is already initialized " << std::endl;
    }

    if (default_device_num < 0)
    {
      std::cout << "Over writing default_device_num " << sycl_default_device_num << std::endl;
      default_device_num = sycl_default_device_num;
    }
  }

  sycl::queue* SYCLDeviceManager::getDefaultDeviceQueuePtr()
  {
    if (sycl_default_device_num < 0)
      throw std::runtime_error("SYCLDeviceManager::getDefaultDeviceQueue() the global instance not initialized.");

#ifdef ENABLE_OFFLOAD
    return visible_devices[sycl_default_device_num].queue_;
#else
    return visible_devices[sycl_default_device_num].queue_.get();
#endif
  }

} // namespace qmcplusplus
