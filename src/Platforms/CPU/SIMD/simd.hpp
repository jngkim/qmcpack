//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/**@file simd.hpp
 *
 * master header file 
 * - inner_product.hpp defines dot, copy and gemv operators
 * - trace.hpp defins trace functions used by determinant classes
 */
#ifndef QMCPLUSPLUS_MATH_SIMD_ADOPTORS_HPP
#define QMCPLUSPLUS_MATH_SIMD_ADOPTORS_HPP

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)

template<typename T>
struct simd_length; // undefined
template<> struct simd_length<float>
{

    static constexpr int value = 
    #if defined(__AVX512F__)
      16;
    #elif defined(__AVX2__) || defined(__AVX__)
      8;
    #else
      4;
    #endif
};
template<> struct simd_length<double>
{
    static constexpr int value = 
    #if defined(__AVX512F__)
      8;
    #elif defined(__AVX2__) || defined(__AVX__)
      4;
    #else
      2;
    #endif
};


/** Emit simdlen(X) omp simd clause
  *\def SIMD_LEN_FOR(TYPENAME)
 *
 *emits omp simd clause "simdlen(X)" where X's value is based on the
 *TYPENAME being float or double and target ISA vector register width.
 *
 */
#   define SIMD_LEN_FOR(TYPENAME) simdlen(simd_length<TYPENAME>::value)
#else
#   define SIMD_LEN_FOR(TYPENAME)
#endif


#include "inner_product.hpp"
#include "vmath.hpp"
#endif
