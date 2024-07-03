//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_COMPLEXVIEW_HPP
#define QMCPLUSPLUS_COMPLEXVIEW_HPP

#ifndef __BYTE_ORDER__ 
  #error Expected __BYTE_ORDER__ to be defined, please modify code below for your target platform
#endif
#ifndef __ORDER_LITTLE_ENDIAN__ 
  #error Expected __ORDER_LITTLE_ENDIAN__ to be defined, please modify code below for your target platform
#endif
#define __QMCPACK_IS_LITTLE_ENDIAN (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)

#include <stdint.h>
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER) 
#include <immintrin.h>
#endif

#include "CPU/SIMD/aligned_allocator.hpp"

#ifdef __cpp_lib_endian
#error YES __cpp_lib_endian is defined!
#endif

namespace qmcplusplus
{

template <typename ToT, typename FromT>
ToT bit_cast (const FromT& val) noexcept
{
     static_assert(sizeof(ToT) == sizeof(FromT),
                   "bit_cast To and From types must be the same size");
     ToT result;
     memcpy ((void *)&result, &val, sizeof(FromT));
     return result;
}

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER) 
// Use compiler intrinsics to implement casts if available, 
// to flow through vectorized loops simpler than memcpy 
template<> float bit_cast<float, uint32_t>(const uint32_t& val) noexcept {
    return _castu32_f32(val);
}
template<> double bit_cast<double, uint64_t>(const uint64_t& val) noexcept {
    return _castu64_f64(val);
}
template<> uint32_t bit_cast<uint32_t, float>(const float& val) noexcept {
    return _castf32_u32(val);
}
template<> uint64_t bit_cast<uint64_t, double>(const double& val) noexcept {
    return _castf64_u64(val);
}
#endif


/** Mimic memory layout of std::complex<T> but allow any type of T
 *\tparam T type of real and imaginary components
  *
 */
template<typename T>
struct ComplexNumber
{
  T re;
  T im;

  constexpr ComplexNumber() = default;

  constexpr ComplexNumber(T re_, T im_)
  : re(re_)
  , im(im_)
  {}

  constexpr ComplexNumber(const ComplexNumber &other)
  : re(other.re)
  , im(other.im)
  {}

  ComplexNumber & operator = (const ComplexNumber &other)
  {
    re = other.re;
    im = other.im;
    return *this;
  }

  // Could add all math ops, or maybe just use std::complex
  constexpr T real() const { return re; }
  constexpr T imag() const { return im; }
  T & real() { return re; }
  T & imag() { return im; }
};


/** View an array of Elements as an array of Complex numbers 
 *\tparam T type of element of the array and Complex Number members
 *
 *Map 2 consecutive array elements to real and imaginary members
 *of a ComplexNumber.  This allows loops to just use the linear
 *index variable to access a ComplexView instead of [2*j] and [2*j+1]
 *which can loose linearity.
 *NOTE: real purpose is to enable ComplexView<float> specialized implementation
 *which can use 64bit unit stride loads|stores to load 2x32bit values. 
 *
 */
template<typename T>
struct ComplexView
{
    using ValueType = typename std::conditional<std::is_const<T>::value, const ComplexNumber<T>, ComplexNumber<T>>::type;
    ValueType * restrict m_complex_ptr;
    explicit ComplexView(T*value_ptr)
    : m_complex_ptr(reinterpret_cast<ValueType *>(value_ptr))
    {
        ASSUME_ALIGNED(m_complex_ptr);
    }

    ValueType & operator[](int index) const {
      return m_complex_ptr[index];
    }
};


// Specialize for Complex float values to take advantage both 32 bit real and
// imaginary overlaying a 64bit value that can be loaded/stored, possibly 
// linearly avoiding strided load/stores with would have been gathers/scatters.
template<>
struct ComplexView<const float>
{
    using StorageType = uint64_t;
    const StorageType * restrict m_complex_ptr;
    explicit ComplexView(const float*value_ptr)
    : m_complex_ptr(reinterpret_cast<const StorageType *>(value_ptr))
    {
        ASSUME_ALIGNED(m_complex_ptr);
    }

    const ComplexNumber<float> operator[](int index) const {
      // Read real and imaginary from memory at same time
      // And hopefully with a linear index
      StorageType stored_val = m_complex_ptr[index];
      float low32bits = bit_cast<float, uint32_t>(static_cast<uint32_t>(stored_val));
      float high32bits = bit_cast<float, uint32_t>(static_cast<uint32_t>(stored_val>>32));
#if __QMCPACK_IS_LITTLE_ENDIAN
      return ComplexNumber<float>{low32bits, high32bits};
#else
      return ComplexNumber<float>{high32bits, low32bits};
#endif
    }
};

// Non const array subscript access returns a Proxy object
// which can be assigned to enabling a 64bit store vs. 2 separate 
// 32bit strided stores which would be scatters.
template<>
struct ComplexView<float>
{
    using StorageType = uint64_t;
    StorageType * restrict m_complex_ptr;
    explicit ComplexView(float*value_ptr)
    : m_complex_ptr(reinterpret_cast<StorageType *>(value_ptr))
    {
        ASSUME_ALIGNED(m_complex_ptr);
    }

    struct ElementProxy {
        ComplexView &m_ca;
        int m_index;
        void operator = (const ComplexNumber<float> &cval)
        {
          // write real and imaginary from memory at same time
          // And hopefully with a linear index
  #if __QMCPACK_IS_LITTLE_ENDIAN
          uint32_t high32bits = bit_cast<uint32_t, float>(cval.imag());
          uint32_t low32bits = bit_cast<uint32_t, float>(cval.real());
  #else
          uint32_t high32bits = bit_cast<uint32_t, float>(cval.real());
          uint32_t low32bits = bit_cast<uint32_t, float>(cval.imag());
  #endif
          StorageType val_to_stored = (static_cast<StorageType>(high32bits)<<32) | static_cast<StorageType>(low32bits);
          m_ca.m_complex_ptr[m_index] = val_to_stored;
        }

        operator const ComplexNumber<float> () const {
          // Read real and imaginary from memory at same time
          // And hopefully with a linear index
          StorageType stored_val = m_ca.m_complex_ptr[m_index];
        float low32bits = bit_cast<float, uint32_t>(static_cast<uint32_t>(stored_val));
        float high32bits = bit_cast<float, uint32_t>(static_cast<uint32_t>(stored_val>>32));
  #if __QMCPACK_IS_LITTLE_ENDIAN
        return ComplexNumber<float>{low32bits, high32bits};
  #else
        return ComplexNumber<float>{high32bits, low32bits};
  #endif
        }
    };
    ElementProxy operator[](int index) {
      return ElementProxy{*this,index};
    }
};


} // namespace qmcplusplus
#endif
