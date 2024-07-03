//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_SPLINE_C2R_HELPER_H
#define QMCPLUSPLUS_SPLINE_C2R_HELPER_H

#include "QMCWaveFunctions/SPOSet.h"

namespace qmcplusplus
{
namespace C2R
{

void assign_vgl_simd(float x, 
                     float y, 
                     float z,
                     SPOSet::ValueVector& psi,
                     SPOSet::GradVector& dpsi,
                     SPOSet::ValueVector& d2psi,
                     const float* myV,
                     const float* myG,
                     const float* myH,
                     int spline_padded_size,
                     const Tensor<float, 3>& G,
                     const Tensor<float, 3>& GGt,
                     const float* myKcart_ptr,
                     const float* mKK,
                     int myKcart_size,
                     int myKcart_padded_size,
                     int first,
                     int last,
                     int nComplexBands);

void assign_vgl_simd(double x, 
                     double y, 
                     double z,
                     SPOSet::ValueVector& psi,
                     SPOSet::GradVector& dpsi,
                     SPOSet::ValueVector& d2psi,
                     const double* myV,
                     const double* myG,
                     const double* myH,
                     int spline_padded_size,
                     const Tensor<double, 3>& G,
                     const Tensor<double, 3>& GGt,
                     const double* myKcart_ptr,
                     const double* mKK,
                     int myKcart_size,
                     int myKcart_padded_size,
                     int first,
                     int last,
                     int nComplexBands);
}
}
#endif
