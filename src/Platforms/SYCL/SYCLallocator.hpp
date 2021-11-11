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
//#include "SYCL/SYCLDeviceManager.hpp"

namespace qmcplusplus
{

extern sycl::queue* get_default_queue();

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
  sycl::queue* m_queue;

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
    return sycl::malloc<T>(n,m_queue->get_device(), m_queue->get_context(),AllocKind);
  }

  void deallocate(T* p, std::size_t)
  {
    sycl::free(p, m_queue->get_context());
  }
};

template<class T1, class T2, sycl::usm::alloc AK1, sycl::usm::alloc AK2>
bool operator==(const SYCLUSMAllocator<T1,AK2>&, const SYCLUSMAllocator<T2,AK2>&)
{
  return false;
}

template<class T1, class T2, sycl::usm::alloc AK1, sycl::usm::alloc AK2>
bool operator!=(const SYCLUSMAllocator<T1,AK2>&, const SYCLUSMAllocator<T2,AK2>&)
{
  return true;
}

template<class T1, class T2, sycl::usm::alloc AK>
bool operator==(const SYCLUSMAllocator<T1,AK>&, const SYCLUSMAllocator<T2,AK>&)
{
  return true;
}

template<class T1, class T2, sycl::usm::alloc AK>
bool operator!=(const SYCLUSMAllocator<T1,AK>&, const SYCLUSMAllocator<T2,AK>&)
{
  return false;
}

template<typename T>
using SYCLSharedAllocator = SYCLUSMAllocator<T,sycl::usm::alloc::shared>;

template<typename T>
using SYCLHostAllocator = SYCLUSMAllocator<T,sycl::usm::alloc::host>;

template<typename T>
using SYCLDeviceAllocator = SYCLUSMAllocator<T,sycl::usm::alloc::device>;

template<typename T>
struct allocator_traits<SYCLHostAllocator<T>>
{
  static const bool is_host_accessible = true;
  static const bool is_dual_space = false;
  static void fill_n(T* ptr, size_t n, const T& value) { std::fill_n(ptr,n,value);}
};

template<typename T>
struct allocator_traits<SYCLSharedAllocator<T>>
{
  static const bool is_host_accessible = true;
  static const bool is_dual_space = true;
  static void fill_n(T* ptr, size_t n, const T& value) { 
    sycl::queue *queue=get_default_queue();
    queue->fill(ptr,value,n).wait();
  }
};

template<typename T>
struct allocator_traits<SYCLDeviceAllocator<T>>
{
  static const bool is_host_accessible =false;
  static const bool is_dual_space = false;
  static void fill_n(T* ptr, size_t n, const T& value) { 
    sycl::queue *queue=get_default_queue();
    queue->fill(ptr,value,n).wait();
  }
};


} // namespace qmcplusplus

#endif
