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


#include <type_traits>

#ifndef QMCPLUSPLUS_ADDRESSING_HPP
#define QMCPLUSPLUS_ADDRESSING_HPP
/**@file Addressing.hpp
 *@brief A collection of functions to improve memory address calculations.
 */

/** Calculate how many bits would be required to represent a 32 bit number
 *\param n number to represent
 *
 *Return how many bits it takes to represent the number N
 *
 */
constexpr uint32_t bits2Represent(uint32_t n) {
  return n <= 1 ? (n == 0 ? 0 : 1) : 1 + bits2Represent((n) / 2);  
}
// Validate bits2Represent
static_assert(bits2Represent(0) == 0, "logic bug");
static_assert(bits2Represent(1) == 1, "logic bug");
static_assert(bits2Represent(2) == 2, "logic bug");
static_assert(bits2Represent(3) == 2, "logic bug");
static_assert(bits2Represent(4) == 3, "logic bug");
static_assert(bits2Represent(5) == 3, "logic bug");

static_assert(bits2Represent(7) == 3, "logic bug");
static_assert(bits2Represent(8) == 4, "logic bug");
static_assert(bits2Represent(9) == 4, "logic bug");

static_assert(bits2Represent(15) == 4, "logic bug");
static_assert(bits2Represent(16) == 5, "logic bug");
static_assert(bits2Represent(17) == 5, "logic bug");

static_assert(bits2Represent(31) == 5, "logic bug");
static_assert(bits2Represent(32) == 6, "logic bug");
static_assert(bits2Represent(33) == 6, "logic bug");

static_assert(bits2Represent(63) == 6, "logic bug");
static_assert(bits2Represent(64) == 7, "logic bug");
static_assert(bits2Represent(65) == 7, "logic bug");

static_assert(bits2Represent(127) == 7, "logic bug");
static_assert(bits2Represent(128) == 8, "logic bug");
static_assert(bits2Represent(129) == 8, "logic bug");

static_assert(bits2Represent(255) == 8, "logic bug");
static_assert(bits2Represent(256) == 9, "logic bug");
static_assert(bits2Represent(257) == 9, "logic bug");


/** Modify index such access array of type ElementT the resulting memory offset
 * fits in 31bits.
 *\tparam ElementT type of array that index intended to dereference
 *\param index index to modify
 *
 *When data access is strided, compiler may assume that
 *INT_MAX*sizeof(ElementT) > INT_MAX and promote the memory
 *access index to 64bit.  We can let compiler know that the index can
 *not be near INT_MAX by masking off enough upper bits to ensure that
 *index*sizeof(ElementT) < INT_MAX and keep 32bit indices.
 *NOTE: the resulting masked index looses its linearity and shouldn't be 
 *used to accesss unit stride arrays as indirect load will happen instead
 *of the expected linear unit stride load!
 *
 */
template<typename ElementT>
__attribute__((always_inline))
int ensure_32bit_offset(int index)
{
  constexpr int bytesRequired = sizeof(ElementT);
  constexpr int maskOffUpperBits = 0x7fffffff>>bits2Represent(bytesRequired);
  return index & maskOffUpperBits;
}


namespace detail {
    struct SameAsInput {};
}
/** Bake any address offsets and type casting into a new base address.
 *\tparam OutElementT Optional Explicit, output element type
 *\tparam InElementT Deduced, input element type
 *\param ptr pointer to rebase
 *
 *To isolate an pointer with offsets from being inlined into 
 *later address calculations, the pointer can be passed through this 
 *non-inlined function.  The Type of the pointer can be also changed.
 *Goal is to establish a new base address and type for memory accesses vs.
 *letting compiler inline previous offsets/casts down into the actual
 *memory accesses.  This can be necessary to get multiple memory
 *accesses to share a common base address as well as keep 64bit bit offset
 *calculations from being combined with 32bit indices used to access the 
 *pointer.
 *
 */
template<typename OutElementT = detail::SameAsInput, typename InElementT>
__attribute__((noinline)) 
typename std::conditional<std::is_same<OutElementT, detail::SameAsInput>::value, InElementT, OutElementT>::type * 
rebasePointer(InElementT *ptr)
{
  return reinterpret_cast<typename std::conditional<std::is_same<OutElementT, detail::SameAsInput>::value, InElementT, OutElementT>::type *>(ptr);
}

#endif
