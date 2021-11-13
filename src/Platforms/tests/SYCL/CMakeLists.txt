#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2021 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////


set(UTEST_EXE test_sycl)
set(UTEST_NAME deterministic-unit_${UTEST_EXE})

add_executable(${UTEST_EXE} test_SYCLallocator.cpp)
target_link_libraries(${UTEST_EXE} platform_runtime containers catch_main)

add_unit_test(${UTEST_NAME} 1 1 $<TARGET_FILE:${UTEST_EXE}>)

set(UTEST_EXE test_sycl_blas)
set(UTEST_NAME deterministic-unit_${UTEST_EXE})

add_executable(${UTEST_EXE} test_syclBLAS.cpp)
target_link_libraries(${UTEST_EXE} catch_main containers)

if(USE_OBJECT_TARGET)
  target_link_libraries(${UTEST_EXE} platform_runtime)
endif()

add_unit_test(${UTEST_NAME} 1 1 $<TARGET_FILE:${UTEST_EXE}>)

set(UTEST_EXE test_sycl_solver)
set(UTEST_NAME deterministic-unit_${UTEST_EXE})

add_executable(${UTEST_EXE} test_syclInverse.cpp)
target_link_libraries(${UTEST_EXE} catch_main containers)

if(USE_OBJECT_TARGET)
  target_link_libraries(${UTEST_EXE} platform_runtime)
endif()

add_unit_test(${UTEST_NAME} 1 1 $<TARGET_FILE:${UTEST_EXE}>)

