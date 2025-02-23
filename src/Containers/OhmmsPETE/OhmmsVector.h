//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////

/** @file
 *
 *  Declaraton of Vector<T,Alloc>
 *  Manage memory through Alloc directly and allow referencing an existing memory.
 */

#ifndef OHMMS_NEW_VECTOR_H
#define OHMMS_NEW_VECTOR_H
#include <algorithm>
#include <vector>
#include <iostream>
#include <type_traits>
#include <stdexcept>
#include "PETE/PETE.h"
#include "allocator_traits.hpp"

namespace qmcplusplus
{
template<class T, typename Alloc = std::allocator<T>>
class Vector
{
public:
  typedef T Type_t;
  typedef T value_type;
  typedef T* iterator;
  typedef const T* const_iterator;
  typedef typename Alloc::size_type size_type;
  typedef typename Alloc::pointer pointer;
  typedef typename Alloc::const_pointer const_pointer;
  typedef Vector<T, Alloc> This_t;

  /** constructor with size n*/
  explicit inline Vector(size_t n = 0, Type_t val = Type_t()) : nLocal(n)
  {
    if (n)
    {
      resize_impl(n);
      qmc_allocator_traits<Alloc>::fill_n(X, n, val);
    }
  }

  /** constructor with an initialized ref */
  explicit inline Vector(T* ref, size_t n) : nLocal(n), X(ref) {}

  /** copy constructor */
  Vector(const Vector& rhs) : nLocal(rhs.nLocal)
  {
    if (nLocal)
    {
      resize_impl(rhs.nLocal);
      if (qmc_allocator_traits<Alloc>::is_host_accessible)
        std::copy_n(rhs.data(), nLocal, X);
      else
        qmc_allocator_traits<Alloc>::fill_n(X, nLocal, T());
    }
  }

  /** This allows construction of a Vector on another containers owned memory that is using a dualspace allocator.
   *  It can be any span of that memory.
   *  You're going to get a bunch of compile errors if the Container in questions is not using a the QMCPACK
   *  realspace dualspace allocator "interface"
   */
  template<typename CONTAINER>
  Vector(CONTAINER& from, T* ref, size_t n) : nLocal(n), X(ref)
  {
    qmc_allocator_traits<Alloc>::attachReference(from.mAllocator, mAllocator, from.data(), ref);
  }

  /** Initializer list constructor that can deal with both POD
   *  and nontrivial nested elements with move assignment operators.
   */
  Vector(std::initializer_list<T> ts)
  {
    if (qmc_allocator_traits<Alloc>::is_host_accessible)
    {
      if (ts.size() == 0)
        return;
      std::size_t num_elements = ts.size();
      resize_impl(num_elements);
      if constexpr (std::is_trivial<T>::value)
      {
        std::copy_n(ts.begin(), ts.size(), X);
      }
      else
      {
        auto ts_it = ts.begin();
        for (int i = 0; i < num_elements; ++i, ++ts_it)
          (*this)[i] = std::move(*ts_it);
      }
    }
    else
      throw std::runtime_error("initializer lists are not supported for Vector's inaccessible to the host");
  }

  // default assignment operator
  inline Vector& operator=(const Vector& rhs)
  {
    if (this == &rhs)
      return *this;
    if (nLocal != rhs.nLocal)
      resize(rhs.nLocal);
    if (qmc_allocator_traits<Alloc>::is_host_accessible)
      std::copy_n(rhs.data(), nLocal, X);
    else
      qmc_allocator_traits<Alloc>::fill_n(X, nLocal, T());
    return *this;
  }

  // assignment operator from anther Vector class
  template<typename T1, typename C1, typename Allocator = Alloc, typename = IsHostSafe<Allocator>>
  inline Vector& operator=(const Vector<T1, C1>& rhs)
  {
    if (std::is_convertible<T1, T>::value)
    {
      if (nLocal != rhs.nLocal)
        resize(rhs.nLocal);
      std::copy_n(rhs.data(), nLocal, X);
    }
    return *this;
  }

  // assigment operator to enable PETE
  template<class RHS, typename Allocator = Alloc, typename = IsHostSafe<Allocator>>
  inline Vector& operator=(const RHS& rhs)
  {
    assign(*this, rhs);
    return *this;
  }

  //! Destructor
  virtual ~Vector() { free(); }

  // Attach to pre-allocated memory
  inline void attachReference(T* ref, size_t n)
  {
    if (nAllocated)
    {
      free();
      // std::cerr << "Allocated OhmmsVector attachReference called.\n" << std::endl;
      // Nice idea but "default" constructed WFC elements in the batched driver make this a mess.
      //throw std::runtime_error("Pointer attaching is not allowed on Vector with allocated memory.");
    }
    nLocal     = n;
    nAllocated = 0;
    X          = ref;
  }

  /** Attach to pre-allocated memory and propagate the allocator of the owning container.
   *  Required for sane access to dual space memory
   */
  template<typename CONTAINER>
  inline void attachReference(const CONTAINER& other, T* ref, size_t n)
  {
    if (nAllocated)
    {
      free();
    }
    nLocal     = n;
    nAllocated = 0;
    X          = ref;
    qmc_allocator_traits<Alloc>::attachReference(other.mAllocator, mAllocator, other.data(), ref);
  }

  //! return the current size
  inline size_t size() const { return nLocal; }

  ///resize
  inline void resize(size_t n, Type_t val = Type_t())
  {
    static_assert(std::is_same<value_type, typename Alloc::value_type>::value,
                  "Vector and Alloc data types must agree!");
    if (nLocal > nAllocated)
      throw std::runtime_error("Resize not allowed on Vector constructed by initialized memory.");

    if (n > nAllocated)
    {
      resize_impl(n);
      qmc_allocator_traits<Alloc>::fill_n(X, n, val);
    }
    else
    {
      if (n > nLocal)
        qmc_allocator_traits<Alloc>::fill_n(X + nLocal, n - nLocal, val);
      nLocal = n;
    }
    return;
  }

  ///clear
  inline void clear() { nLocal = 0; }

  inline void zero() { qmc_allocator_traits<Alloc>::fill_n(X, nAllocated, T()); }

  ///free
  inline void free()
  {
    if (nAllocated)
    {
      mAllocator.deallocate(X, nAllocated);
    }
    nLocal     = 0;
    nAllocated = 0;
    X          = nullptr;
  }

  // Get and Set Operations
  template<typename Allocator = Alloc, typename = IsHostSafe<Allocator>>
  inline Type_t& operator[](size_t i)
  {
    return X[i];
  }

  template<typename Allocator = Alloc, typename = IsHostSafe<Allocator>>
  inline const Type_t& operator[](size_t i) const
  {
    return X[i];
  }

  //inline Type_t& operator()(size_t i)
  //{
  //  return X[i];
  //}

  //inline Type_t operator()( size_t i) const
  //{
  //  return X[i];
  //}

  inline iterator begin() { return X; }
  inline const_iterator begin() const { return X; }

  inline iterator end() { return X + nLocal; }
  inline const_iterator end() const { return X + nLocal; }

  inline pointer data() { return X; }
  inline const_pointer data() const { return X; }

  /** Return the device_ptr matching X if this is a vector attached or
   *  owning dual space memory.
   */
  template<typename Allocator = Alloc, typename = IsDualSpace<Allocator>>
  inline pointer device_data()
  {
    return mAllocator.get_device_ptr();
  }
  template<typename Allocator = Alloc, typename = IsDualSpace<Allocator>>
  inline const_pointer device_data() const
  {
    return mAllocator.get_device_ptr();
  }

  inline pointer first_address() { return X; }
  inline const_pointer first_address() const { return X; }

  inline pointer last_address() { return X + nLocal; }
  inline const_pointer last_address() const { return X + nLocal; }

  // Abstract Dual Space Transfers
  template<typename Allocator = Alloc, typename = IsDualSpace<Allocator>>
  void updateTo()
  {
    qmc_allocator_traits<Alloc>::updateTo(mAllocator, X, nLocal);
  }
  template<typename Allocator = Alloc, typename = IsDualSpace<Allocator>>
  void updateFrom()
  {
    qmc_allocator_traits<Alloc>::updateFrom(mAllocator, X, nLocal);
  }

private:
  ///size
  size_t nLocal = 0;
  ///The number of allocated
  size_t nAllocated = 0;
  ///pointer to the data accessed through this object
  T* X = nullptr;
  ///allocator
  Alloc mAllocator;

  ///a dumb resize, always free existing memory and resize to n. n must be protected positive
  inline void resize_impl(size_t n)
  {
    if (nAllocated)
    {
      mAllocator.deallocate(X, nAllocated);
    }
    X          = mAllocator.allocate(n);
    nLocal     = n;
    nAllocated = n;
  }
};

} // namespace qmcplusplus

#include "OhmmsPETE/OhmmsVectorOperators.h"

namespace qmcplusplus
{
//-----------------------------------------------------------------------------
// We need to specialize CreateLeaf<T> for our class, so that operators
// know what to stick in the leaves of the expression tree.
//-----------------------------------------------------------------------------
template<class T, class C>
struct CreateLeaf<Vector<T, C>>
{
  typedef Reference<Vector<T, C>> Leaf_t;
  inline static Leaf_t make(const Vector<T, C>& a) { return Leaf_t(a); }
};

//-----------------------------------------------------------------------------
// We need to write a functor that is capable of comparing the size of
// the vector with a stored value. Then, we supply LeafFunctor specializations
// for Scalar<T> and STL vector leaves.
//-----------------------------------------------------------------------------
class SizeLeaf
{
public:
  SizeLeaf(int s) : size_m(s) {}
  SizeLeaf(const SizeLeaf& model) : size_m(model.size_m) {}
  bool operator()(int s) const { return size_m == s; }

private:
  int size_m;
};

template<class T>
struct LeafFunctor<Scalar<T>, SizeLeaf>
{
  typedef bool Type_t;
  inline static bool apply(const Scalar<T>&, const SizeLeaf&)
  {
    // Scalars always conform.
    return true;
  }
};

template<class T, class C>
struct LeafFunctor<Vector<T, C>, SizeLeaf>
{
  typedef bool Type_t;
  inline static bool apply(const Vector<T, C>& v, const SizeLeaf& s) { return s(v.size()); }
};

//-----------------------------------------------------------------------------
// EvalLeaf1 is used to evaluate expression with vectors.
// (It's already defined for Scalar values.)
//-----------------------------------------------------------------------------

template<class T, class C>
struct LeafFunctor<Vector<T, C>, EvalLeaf1>
{
  typedef T Type_t;
  inline static Type_t apply(const Vector<T, C>& vec, const EvalLeaf1& f) { return vec[f.val1()]; }
};

//////////////////////////////////////////////////////////////////////////////////
// LOOP is done by evaluate function
//////////////////////////////////////////////////////////////////////////////////
template<class T, class C, class Op, class RHS>
inline void evaluate(Vector<T, C>& lhs, const Op& op, const Expression<RHS>& rhs)
{
  if (forEach(rhs, SizeLeaf(lhs.size()), AndCombine()))
  {
    // We get here if the vectors on the RHS are the same size as those on
    // the LHS.
    for (int i = 0; i < lhs.size(); ++i)
    {
      // The actual assignment operation is performed here.
      // PETE operator tags all define operator() to perform the operation.
      // (In this case op performs an assignment.) forEach is used
      // to compute the rhs value.  EvalLeaf1 gets the
      // values at each node using random access, and the tag
      // OpCombine tells forEach to use the operator tags in the expression
      // to combine values together.
      op(lhs[i], forEach(rhs, EvalLeaf1(i), OpCombine()));
    }
  }
  else
  {
    throw std::runtime_error("Error in evaluate: LHS and RHS don't conform in OhmmsVector.");
  }
}

template<class T, class Alloc>
bool operator==(const Vector<T, Alloc>& lhs, const Vector<T, Alloc>& rhs)
{
  static_assert(qmc_allocator_traits<Alloc>::is_host_accessible, "operator== requires host accessible Vector.");
  if (lhs.size() == rhs.size())
  {
    for (int i = 0; i < rhs.size(); i++)
      if (lhs[i] != rhs[i])
        return false;
    return true;
  }
  else
    return false;
}

template<class T, class Alloc>
bool operator!=(const Vector<T, Alloc>& lhs, const Vector<T, Alloc>& rhs)
{
  static_assert(qmc_allocator_traits<Alloc>::is_host_accessible, "operator== requires host accessible Vector.");
  return !(lhs == rhs);
}

// I/O
template<class T, class Alloc>
std::ostream& operator<<(std::ostream& out, const Vector<T, Alloc>& rhs)
{
  static_assert(qmc_allocator_traits<Alloc>::is_host_accessible, "operator<< requires host accessible Vector.");
  for (int i = 0; i < rhs.size(); i++)
    out << rhs[i] << std::endl;
  return out;
}

template<class T, class Alloc>
std::istream& operator>>(std::istream& is, Vector<T, Alloc>& rhs)
{
  static_assert(qmc_allocator_traits<Alloc>::is_host_accessible, "operator>> requires host accessible Vector.");
  //printTinyVector<TinyVector<T,D> >::print(out,rhs);
  for (int i = 0; i < rhs.size(); i++)
    is >> rhs[i];
  return is;
}

} // namespace qmcplusplus

#endif // OHMMS_PARTICLEATTRIB_PEPE_H
