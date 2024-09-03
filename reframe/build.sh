#!/bin/bash

ml unload oneapi/eng-compiler/2024.04.15.002
module use -a /soft/preview/pe/24.180.0-RC5/modulefiles
ml load oneapi/release/2024.2.1
ml load spack-pe-gcc/0.7.0-24.086.0
ml load gcc/12.2.0
export HDF5_ROOT=/opt/aurora/24.086.0/spack/oneapi/0.7.0/install/2024.04.15.002/linux-sles15-x86_64/oneapi-2024.04.15.002/hdf5-1.14.3-e25jqi66pc7j3o26lt57ijpzqmabubvl
ml load cmake

src_dir=qmcpack   # topdir of source
build_dir=build   # build directory

CXX=mpicxx CC=icx ${CMAKE} \
  -S ${src_dir} \
  -B ${build_dir} \
  -DENABLE_SYCL=ON \
  -DENABLE_OFFLOAD=ON \
  -DCMAKE_CXX_FLAGS="-mprefer-vector-width=512  -march=sapphirerapids" \
  -DCMAKE_C_FLAGS="-mprefer-vector-width=512  -march=sapphirerapids" \
  -DENABLE_PHDF5=ON \
  -DQMC_MIXED_PRECISION=ON 
