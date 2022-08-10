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


build_dir=build_ompt
log_file=log.${build_dir}.${mtag}.txt

report_system ${log_file}

CXX=icpx CC=icx cmake \
  -S ${src_dir} \
  -B ${build_dir} \
  -DLibXml2_ROOT=${LIB_DIR} \
  -DHDF5_ROOT=${LIB_DIR} \
  -DENABLE_OFFLOAD=ON \
  -DOFFLOAD_TARGET=spir64 \
  -DQMC_MPI=OFF \
  -DQMC_MIXED_PRECISION=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DUSE_OBJECT_TARGET=ON 2>&1 | tee -a ${log_file}


#cmake --build ${build_dir} --parallel 32 2>&1 | tee -a ${log_file}
