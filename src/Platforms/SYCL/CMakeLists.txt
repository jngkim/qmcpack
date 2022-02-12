#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2021 QMCPACK developers.
#//
#// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp 
#//
#// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp
#//////////////////////////////////////////////////////////////////////////////////////


set(SYCL_RT_SRCS SYCLruntime.cpp SYCLallocator.cpp)

if(USE_OBJECT_TARGET)
  add_library(platform_sycl_runtime OBJECT ${SYCL_RT_SRCS})
else()
  add_library(platform_sycl_runtime ${SYCL_RT_SRCS})
endif()

find_library(MKL_SYCL mkl_sycl
  HINTS ${MKL_ROOT} $ENV{MKLROOT} $ENV{MKL_ROOT} $ENV{MKL_HOME}
  PATH_SUFFIXES lib/intel64
  )
target_link_libraries(platform_sycl_runtime PUBLIC ${MKL_SYCL} ${MKL_LIBRARIES})
