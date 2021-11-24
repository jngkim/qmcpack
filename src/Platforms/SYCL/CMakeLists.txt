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

# temporary need MKL::SYCL
target_link_libraries(platform_sycl_runtime PUBLIC "-L$ENV{MKLROOT}/lib/intel64 -lmkl_sycl -lsycl -lOpenCL")


#CXX=icpx CC=icx cmake ~/workspace/GitHub/qmcpack.mkl \
#-DCMAKE_CXX_FLAGS="-g -xCORE-AVX512 -mprefer-vector-width=512 -DMKL_ILP64 -DENABLE_MKL_OFFLOAD -fsycl" \
#-DCMAKE_C_FLAGS="-g -xCORE-AVX512 -mprefer-vector-width=512" \
#-DENABLE_OFFLOAD=1 -DOFFLOAD_TARGET=spir64 \
#-DUSE_OBJECT_TARGET=ON \
#-DQMC_MIXED_PRECISION=1 \
#-DBUILD_LMYENGINE_INTERFACE=OFF \
#-DQMC_MPI=0 \
#-DBLA_VENDOR=Intel10_64ilp

