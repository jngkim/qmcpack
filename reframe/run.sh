#!/bin/bash
export NEOReadDebugKeys=1
export SplitBcsCopy=0
export UR_L0_SERIALIZE=2

export KMP_BLOCKTIME=0
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export HYDRA_TOPO_DEBUG=1

export MPIR_CVAR_ENABLE_GPU=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
unset MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE 
unset MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE 
unset MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE

#LAMMPS debug session
export FI_MR_CACHE_MONITOR=disabled
export FI_MR_ZE_CACHE_MONITOR_ENABLED=0
export MPIR_CVAR_CH4_ROOTS_ONLY_PMI=1
export PALS_PMI=pmix

export OMP_TARGET_OFFLOAD=MANDATORY
export LIBOMPTARGET_PLUGIN=LEVEL0
export LIBOMP_USE_HIDDEN_HELPER_TASK=0
export LIBOMP_NUM_HIDDEN_HELPER_THREADS=0

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

export LIBOMPTARGET_LEVEL_ZERO_COMPILATION_OPTIONS="-ze-opt-large-register-file"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"

export LIBOMPTARGET_LEVEL_ZERO_COMMAND_MODE=sync
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

export MPI_BIND_OPTIONS="--cpu-bind list:1-8,105-112:9-16,113-120:17-24,121-128:27-34,131-138:35-42,139-146:43-50,147-154:53-60,157-164:61-68,165-172:69-76,173-180:79-86,183-190:87-94,191-198:95-102,199-206"


ppn=12
omp=8

mpi=$(( ppn * nnodes ))

# Note
qmcpack=<binary>
input=<input xml>

timeout 1200 mpiexec --hostfile hostfile \
  -np ${mpi} -ppn 12 --pmi=pmix -genv OMP_NUM_THREADS=${omp} $MPI_BIND_OPTIONS \
  ./gpu_mapper.sh \
  ${qmcpack} ${input} \
  --enable-timers=fine 

