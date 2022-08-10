#!/bin/bash
TARGET_PLATFORM=${1-gpu}

src_dir=.
LIB_DIR=/nfs/pdx/home/jeongnim/workspace/build/applications.hpc.workloads.aurora.qmcpack/metal-env/share

mtag=`date "+%Y%m%d.%H%M"`

report_system() {
  uname -r &> $1
  clang++  --version &>> $1
  cuda --version &>> $1
  echo $MKLROOT >> $1
}


build_dir=build_ompt_hip
log_file=log.${build_dir}.${mtag}.txt

report_system ${log_file}

CXX=clang++ CC=clang cmake \
  -S ${src_dir} \
  -B ${build_dir} \
  -DLibXml2_ROOT=${LIB_DIR} \
  -DHDF5_ROOT=${LIB_DIR} \
  -DCMAKE_CUDA_COMPILER=clang++ \
  -DENABLE_OFFLOAD=ON \
  -DQMC_MPI=OFF \
  -DQMC_MIXED_PRECISION=ON \
  -DENABLE_CUDA=ON \
  -DQMC_ENABLE_CUDA=ON \
  -DQMC_CUDA2HIP=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DUSE_OBJECT_TARGET=ON 2>&1 | tee -a ${log_file}


#cmake --build ${build_dir} --parallel 32 2>&1 | tee -a ${log_file}
