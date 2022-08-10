#!/bin/bash
TARGET_PLATFORM=${1-gpu}

src_dir=.

mtag=`date "+%Y%m%d.%H%M"`

report_system() {
  uname -r &> $1
  clang++  --version &>> $1
  cuda --version &>> $1
  echo $MKLROOT >> $1
}


build_dir=build_cpu
log_file=log.${build_dir}.${mtag}.txt

report_system ${log_file}

CXX=clang++ CC=clang cmake \
  -S ${src_dir} \
  -B ${build_dir} \
  -DQMC_MPI=OFF \
  -DQMC_MIXED_PRECISION=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  2>&1 | tee -a ${log_file}


#cmake --build ${build_dir} --parallel 32 2>&1 | tee -a ${log_file}
