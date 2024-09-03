//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include "SYCLDeviceManager.h"
#include "SYCLruntime.hpp"

namespace qmcplusplus
{
sycl::queue& getSYCLDefaultDeviceDefaultQueue() { return SYCLDeviceManager::getDefaultDeviceDefaultQueue(); }

sycl::queue createSYCLInOrderQueueOnDefaultDevice()
{
  return sycl::queue(getSYCLDefaultDeviceDefaultQueue().get_context(), getSYCLDefaultDeviceDefaultQueue().get_device(),
                     sycl::property::queue::in_order());
}
sycl::queue& getSYCLInOrderQueueOnDefaultDevice()
{
  static std::vector<sycl::queue*> queue_pool;
  if(queue_pool.empty())
  {
    queue_pool.resize(omp_get_max_threads());
    queue_pool[0] = &getSYCLDefaultDeviceDefaultQueue();
    for(int i=1; i<queue_pool.size(); ++i)
      queue_pool[i] = new sycl::queue{getSYCLDefaultDeviceDefaultQueue().get_context(), 
          getSYCLDefaultDeviceDefaultQueue().get_device(), 
          sycl::property::queue::in_order()};
  }
  int iq = omp_get_thread_num() % queue_pool.size();
  return *queue_pool[iq];
}

sycl::queue createSYCLQueueOnDefaultDevice()
{
  return sycl::queue(getSYCLDefaultDeviceDefaultQueue().get_context(), getSYCLDefaultDeviceDefaultQueue().get_device());
}

size_t getSYCLdeviceFreeMem()
{
  auto device = getSYCLDefaultDeviceDefaultQueue().get_device();
  if (device.has(sycl::aspect::ext_intel_free_memory))
    return getSYCLDefaultDeviceDefaultQueue().get_device().get_info<sycl::ext::intel::info::device::free_memory>();
  else
    return 0;
}
} // namespace qmcplusplus
