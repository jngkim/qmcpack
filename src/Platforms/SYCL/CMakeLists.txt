#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2020 QMCPACK developers.
#//
#// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp 
#//
#// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp
#//////////////////////////////////////////////////////////////////////////////////////


set(SYCL_RT_SRCS SYCLDeviceManager.cpp)

if(USE_OBJECT_TARGET)
  add_library(platform_sycl_runtime OBJECT ${SYCL_RT_SRCS})
else()
  add_library(platform_sycl_runtime ${SYCL_RT_SRCS})
endif()

# temporary
target_link_libraries(platform_sycl_runtime PUBLIC "-L$ENV{MKLROOT}/lib/intel64 -lmkl_sycl -lsycl -lOpenCL")
