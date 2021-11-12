//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Thomas Applencourt, tappencourt@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp
//////////////////////////////////////////////////////////////////////////////////////


#include <cstddef>
#include <atomic>
#include <map>
#include "SYCLDeviceManager.hpp"
#include <level_zero/ze_api.h>
#include <CL/sycl/backend/level_zero.hpp>
#include <omp.h>

namespace qmcplusplus
{
  // General case where each Context can have target multiple Device
  std::vector<struct syclDeviceInfo> xomp_get_infos_devices() {
    static std::vector<struct syclDeviceInfo> ompDeviceId2Context;
    if (!ompDeviceId2Context.empty())
      return ompDeviceId2Context;

    // This datascructure is a vector used to map a OpenMP device ID to a sycl context and device
    ompDeviceId2Context.resize(omp_get_num_devices());

    //1. Map each level zero (platform,context) to a vector a sycl::device.
    //   This is requied to create SYCL::context spaming multiple devices.
    //   We use a map implace of a inordermap to avoid some implicitly-deleted default constructor error
    std::map<std::pair<ze_driver_handle_t,ze_context_handle_t>, std::vector<sycl::device>> hContext2device;
    for (int D=0; D< omp_get_num_devices(); D++) {
      omp_interop_t o = 0;
#pragma omp interop init(targetsync: o) device(D)
      int err = -1;
      const ze_driver_handle_t hPlatform = static_cast<ze_driver_handle_t>(omp_get_interop_ptr(o, omp_ipr_platform, &err));
      assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_platform)");
      const ze_context_handle_t hContext = static_cast<ze_context_handle_t>(omp_get_interop_ptr(o, omp_ipr_device_context, &err));
      assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_device_context)");
      // equivalent to:
      //ze_context_handle_t hContext = static_cast<ze_context_handle_t>(omp_target_get_context(0));
      const ze_device_handle_t hDevice =  static_cast<ze_device_handle_t>(omp_get_interop_ptr(o, omp_ipr_device, &err));
      assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_device)");
#pragma omp interop destroy(o)

      // Store the Level_zero context. This will be required to create the SYCL context latter
      ompDeviceId2Context[D].ze_context = hContext;

      const sycl::platform sycl_platform = sycl::level_zero::make<sycl::platform>(hPlatform);

      ompDeviceId2Context[D].sycl_device = sycl::level_zero::make<sycl::device>(sycl_platform, hDevice);
      hContext2device[std::make_pair(hPlatform,hContext)].push_back(ompDeviceId2Context[D].sycl_device);
    }

    // Construct sycl::contexts who stawn multiple openmp device, if possible.
    // This is N2, but trivial to make it log(N)
    for ( const auto& [ hPlatforContext, sycl_devices]: hContext2device ) {
      const auto& [ _, hContext] = hPlatforContext;
      // This only work because the backend poiter is saved as a shared_pointer in SYCL context with Intel Implementation
      // https://github.com/intel/llvm/blob/ef33c57e48237c7d918f5dab7893554cecc001dd/sycl/source/backend/level_zero.cpp#L59
      // As far as I know this is not required by the SYCL2020 Spec
      const sycl::context sycl_context = sycl::level_zero::make<sycl::context>(sycl_devices, hContext,  sycl::level_zero::ownership::keep);

      for (int D=0; D< omp_get_num_devices(); D++)
        if (ompDeviceId2Context[D].ze_context == hContext)
          ompDeviceId2Context[D].sycl_context = sycl_context;
    }

    std::cout << "Number of Contexts " << ompDeviceId2Context.size() << std::endl;
    return ompDeviceId2Context;
  }

  //singleton 
  sycl::queue* get_default_queue()
  {
    static sycl::queue* global_queue_danger=nullptr;
    if(global_queue_danger==nullptr)
    {
      auto xomp_devices=xomp_get_infos_devices();
      global_queue_danger=
        new sycl::queue{xomp_devices[0].sycl_context, xomp_devices[0].sycl_device};
    }
    return global_queue_danger;
  }
}
