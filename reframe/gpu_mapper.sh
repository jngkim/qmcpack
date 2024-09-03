#!/bin/bash                                                                                                                                      

# Original author: Vikram Narayana (vikram.narayana@intel.com)

LOCAL_RANK_ID=${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID}}
export ZE_AFFINITY_MASK=${LOCAL_RANK_ID}

#echo 'ZE_AFFINITY_MASK on '`hostname`' for local rank '${LOCAL_RANK_ID}' is '${ZE_AFFINITY_MASK}

# Invoke the main program
$*
