//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file
 */
#ifndef QMCPLUSPLUS_SYCL_ALLOCATOR_H
#define QMCPLUSPLUS_SYCL_ALLOCATOR_H

#include <memory>
#include <type_traits>
#include <atomic>
#include "config.h"
#include "allocator_traits.hpp"

#include <omp.h>
#include <CL/sycl.hpp>

namespace qmcplusplus
{

extern sycl::queue* get_default_queue();

extern std::atomic<size_t> OMPallocator_device_mem_allocated;

inline size_t getOMPdeviceMemAllocated() { return OMPallocator_device_mem_allocated; }

template<typename T>
T* getOffloadDevicePtr(T* host_ptr)
{
  T* device_ptr;
  PRAGMA_OFFLOAD("omp target data use_device_ptr(host_ptr)") { device_ptr = host_ptr; }
  return device_ptr;
}

/** OMPallocator is an allocator with fused device and dualspace allocator functionality.
 *  it is mostly c++03 style but is stateful with respect to the bond between the returned pt on the host
 *  and the device_ptr_.  While many containers may need a copy only one can own the memory
 *  it returns and it can only service one owner. i.e. only one object should call the allocate
 *  and deallocate methods.
 *
 *  Note: in the style of openmp portability this class always thinks its dual space even when its not,
 *  this happens through the magic of openmp ignoring target pragmas when target isn't enabled. i.e.
 *  -fopenmp-targets=... isn't passed at compile time.
 *  This makes the code "simpler" and more "portable" since its the same code you would write for
 *  openmp CPU implementation *exploding head* and that is the same implementation + pragmas
 *  as the serial implementation. This definitely isn't true for all QMCPACK code using offload
 *  but it is true for OMPAllocator so we do test it that way.
 */
template<typename T, class HostAllocator = std::allocator<T>>
struct OMPallocator : public HostAllocator
{
  using value_type    = typename HostAllocator::value_type;
  using size_type     = typename HostAllocator::size_type;
  using pointer       = typename HostAllocator::pointer;
  using const_pointer = typename HostAllocator::const_pointer;

  OMPallocator() = default;
  /** Gives you a OMPallocator with no state.
   *  But OMPallocoator is stateful so this copy constructor is a lie.
   *  However until allocators are correct > c++11 this is retained since
   *  our < c++11 compliant containers may expect it.
   */
  OMPallocator(const OMPallocator&) : device_ptr_{nullptr}, omp_queue{nullptr} {}
  template<class U, class V>
  OMPallocator(const OMPallocator<U, V>&) : device_ptr_{nullptr}, omp_queue{nullptr}
  {}

  template<class U, class V>
  struct rebind
  {
    typedef OMPallocator<U, V> other;
  };

  value_type* allocate(std::size_t n)
  {
    static_assert(std::is_same<T, value_type>::value, "OMPallocator and HostAllocator data types must agree!");
    if(omp_queue==nullptr) omp_queue=get_default_queue();
    value_type* pt = HostAllocator::allocate(n);
    device_ptr_ =  sycl::malloc_device<T>(n,*omp_queue);
    const int status = omp_target_associate_ptr(pt, device_ptr_, n * sizeof(T), 0, omp_get_default_device());
    OMPallocator_device_mem_allocated += n * sizeof(T);
    return pt;
  }

  void deallocate(value_type* pt, std::size_t n)
  {
    OMPallocator_device_mem_allocated -= n * sizeof(T);
    if(omp_queue==nullptr) 
      throw std::runtime_error("omp_queue is uninitialized");
    //T* device_ptr_from_omp = getOffloadDevicePtr(pt);
    const int status = omp_target_disassociate_ptr(pt, omp_get_default_device());
    sycl::free(device_ptr_,omp_queue->get_context());
    HostAllocator::deallocate(pt, n);
  }

  void attachReference(const OMPallocator& from, std::ptrdiff_t ptr_offset)
  {
    device_ptr_ = const_cast<typename OMPallocator::pointer>(from.get_device_ptr()) + ptr_offset;
  }

  T* get_device_ptr() { return device_ptr_; }
  const T* get_device_ptr() const { return device_ptr_; }

  // pointee is on device.
  T* device_ptr_ = nullptr;
  sycl::queue* omp_queue=nullptr;
};

/** Specialization for OMPallocator which is a special DualAllocator with fused
 *  device and dualspace allocator functionality.
 */
template<typename T, class HostAllocator>
struct qmc_allocator_traits<OMPallocator<T, HostAllocator>>
{
  static const bool is_host_accessible = true;
  static const bool is_dual_space      = true;

  static void fill_n(T* ptr, size_t n, const T& value)
  {
    qmc_allocator_traits<HostAllocator>::fill_n(ptr, n, value);
    //PRAGMA_OFFLOAD("omp target update to(ptr[:n])")
  }

  static void attachReference(const OMPallocator<T, HostAllocator>& from,
                              OMPallocator<T, HostAllocator>& to,
                              const T* from_data,
                              T* ref)
  {
    std::ptrdiff_t ptr_offset = ref - from_data;
    to.attachReference(from, ptr_offset);
  }

  static void updateTo(OMPallocator<T, HostAllocator>& alloc, T* host_ptr, size_t n)
  {
    alloc.omp_queue->memcpy(alloc.get_device_ptr(),host_ptr,n*sizeof(T)).wait();
  }

  static void updateFrom(OMPallocator<T, HostAllocator>& alloc, T* host_ptr, size_t n)
  {
    alloc.omp_queue->memcpy(host_ptr,alloc.get_device_ptr(),n*sizeof(T)).wait();
  }

  static void deviceSideCopyN(OMPallocator<T, HostAllocator>& alloc, size_t to, size_t n, size_t from)
  {
    auto* dev_ptr = alloc.get_device_ptr();
    alloc.omp_queue->memcpy(dev_ptr+to,dev_ptr+from,n*sizeof(T)).wait();
  }
};
} // namespace qmcplusplus
#endif
