//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file SYCLallocator.hpp
 * this file provides three C++ memory allocators using SYCL specific memory allocation functions.
 *
 * SYCLManagedAllocator allocates SYCL unified memory
 * SYCLAllocator allocates SYCL device memory
 * SYCLHostAllocator allocates SYCL host pinned memory
 */
#ifndef QMCPLUSPLUS_SYCL_ALLOCATOR_H
#define QMCPLUSPLUS_SYCL_ALLOCATOR_H

#include <cstdlib>
#include <stdexcept>
#include <CL/sycl.hpp>
#include "config.h"
#include "allocator_traits.hpp"
#include "SYCLDeviceManager.hpp"

namespace qmcplusplus
{

//extern sycl::queue* get_default_queue();

/** allocator for SYCL unified memory
 * @tparam T data type
 * @tparam AllocKind sycl::sum::alloc
 */
template<typename T, sycl::usm::alloc AllocKind>
struct SYCLUSMAllocator
{
  typedef T value_type;
  typedef size_t size_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  sycl::queue* m_queue=nullptr;
  // pointee is on device.
  T* device_ptr_ = nullptr;

  SYCLUSMAllocator():m_queue{get_default_queue()} 
  {
  }

  template<class U>
  SYCLUSMAllocator(const SYCLUSMAllocator<U,AllocKind>&)
  {}

  template<class U>
  struct rebind
  {
    typedef SYCLUSMAllocator<U,AllocKind> other;
  };

  T* allocate(std::size_t n)
  {
    //return sycl::aligned_alloc<T>(QMC_SIMD_ALIGNMENT,n,m_queue.get_device(), m_queue.get_context(),AllocKind);
    if(m_queue == nullptr) m_queue=get_default_queue();
    device_ptr_=sycl::malloc<T>(n,m_queue->get_device(), m_queue->get_context(),AllocKind);
    return device_ptr_;
  }

  void deallocate(T* p, std::size_t)
  {
    if(p == device_ptr_)
      sycl::free(p, m_queue->get_context());
    device_ptr_=nullptr;
  }

  T* get_device_ptr() { return device_ptr_; }
  const T* get_device_ptr() const { return device_ptr_; }

};

template<typename T>
using SYCLSharedAllocator = SYCLUSMAllocator<T,sycl::usm::alloc::shared>;

template<typename T>
using SYCLHostAllocator = SYCLUSMAllocator<T,sycl::usm::alloc::host>;

template<typename T>
using SYCLDeviceAllocator = SYCLUSMAllocator<T,sycl::usm::alloc::device>;


/** host/shared traits **/
template<typename T, sycl::usm::alloc AllocKind>
struct qmc_allocator_traits<SYCLUSMAllocator<T,AllocKind>>
{
  static const bool is_host_accessible = true;
  static const bool is_dual_space      = true;

  static void fill_n(T* ptr, size_t n, const T& value)
  {
    std::fill_n(ptr, n, value);
  }

  static void attachReference(const SYCLUSMAllocator<T,AllocKind>& from,
                              SYCLUSMAllocator<T,AllocKind>& to,
                              const T* from_data,
                              T* ref)
  {
    std::ptrdiff_t ptr_offset = ref - from_data;
    to.attachReference(from, ptr_offset);
  }

  static void updateTo(SYCLSharedAllocator<T>& alloc, T* host_ptr, size_t n)
  {
    std::cout << "Do nothing with updateTo" << std::endl;
  }

  static void updateFrom(SYCLSharedAllocator<T>& alloc, T* host_ptr, size_t n)
  {
    std::cout << "Do nothing with updateFrom" << std::endl;
  }

  // Not very optimized device side copy.  Only used for testing.
  static void deviceSideCopyN(SYCLSharedAllocator<T>& alloc, size_t to, size_t n, size_t from)
  {
  }
};

/** device traits **/
template<typename T>
struct qmc_allocator_traits<SYCLDeviceAllocator<T>>
{
  static const bool is_host_accessible = false;
  static const bool is_dual_space      = false;

  static void attachReference(const SYCLDeviceAllocator<T>& from,
                              SYCLDeviceAllocator<T>& to,
                              const T* from_data,
                              T* ref)
  {
    std::ptrdiff_t ptr_offset = ref - from_data;
    to.attachReference(from, ptr_offset);
  }

  static void updateTo(SYCLDeviceAllocator<T>& alloc, T* host_ptr, size_t n)
  {
    std::cout << "Illegal updateTo" << std::endl;
  }

  static void updateFrom(SYCLDeviceAllocator<T>& alloc, T* host_ptr, size_t n)
  {
    std::cout << "Illegal updateFrom" << std::endl;
  }

  // Not very optimized device side copy.  Only used for testing.
  static void deviceSideCopyN(SYCLDeviceAllocator<T>& alloc, size_t to, size_t n, size_t from)
  {
  }

};



} // namespace qmcplusplus

#endif
