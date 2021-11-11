//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_OMPBLAS_DPCPP_INTEROP_H
#define QMCPLUSPLUS_OMPBLAS_DPCPP_INTEROP_H

#include <complex>
#include <CL/sycl.hpp>
#include "mkl.h"
#include "oneapi/mkl/blas.hpp"

namespace qmcplusplus
{

//extern sycl::queue* get_default_queue();

/** Implement selected batched and non-batched BLAS2 calls using OpenMP offload for different data types S/C/D/Z
 * 1) column major like the BLAS fortran API
 * 2) all the functions are synchronous, expected to be changed to asynchronous in the future.
 * 3) all the po_inter arguments are expected as device poompBLAS_inters.
 * 4) in batched APIs, alpha and beta are **not** scalars but po_inters to array of batch size.
 */
namespace syclBLAS
{

typedef std::int64_t syclBLAS_int;
typedef sycl::event syclBLAS_status;
typedef sycl::queue syclBLAS_handle;

template<typename T>
inline sycl::event gemv(sycl::queue&   handle,
                    const oneapi::mkl::transpose trans,
                    const int          m,
                    const int          n,
                    const T            alpha,
                    const T* const     A,
                    const int          lda,
                    const T* const     x,
                    const int          incx,
                    const T            beta,
                    T* const           y,
                    const int          incy)
{
  return oneapi::mkl::blas::gemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y,incy);
}

template<typename T>
inline sycl::event gemv_batched(sycl::queue&   handle,
                    const oneapi::mkl::transpose trans,
                    const syclBLAS_int  m,
                    const syclBLAS_int  n,
                    const T*            alpha,
                    const T**           A,
                    const syclBLAS_int  lda,
                    const T**           x,
                    const syclBLAS_int  incx,
                    const T*            beta,
                    T**                 y,
                    const syclBLAS_int  incy,
                    const syclBLAS_int  batch_count)
{
  //using group API: only one group
  return oneapi::mkl::blas::gemv_batch(handle, &trans, &m, &n, alpha, A, &lda,x,&incx, beta, y, &incy,1,&batch_count);
}

template<typename T>
inline sycl::event ger(sycl::queue& handle,
                       const unsigned  m,
                       const unsigned  n,
                       const T         alpha,
                       const T* const  x,
                       const unsigned  incx,
                       const T* const  y,
                       const unsigned  incy,
                       T* const        A,
                       const unsigned  lda)
{
  constexpr size_t ts=16;

  const size_t m_max=((m+ts-1)/ts)*ts;
  const size_t n_max=((n+ts-1)/ts)*ts;

  return handle.submit([&](handler& cgh) {
      cgh.parallel_for(nd_range<2>{{m_max,n_max},{ts,ts}},
          [=](nd_item<2> item) { // [[cl::intel_reqd_sub_group_size(32)]] {
          unsigned x_g = item.get_global_id(0);
          unsigned y_g = item.get_global_id(1);
          if(x_g<m && y_g<n) 
          A[x_g*lda + y_g] += alpha*x[y_g]*y[x_g];
          });
      });
}

template<typename T>
inline sycl::event ger_batched(sycl::queue& handle,
                                const int       m,
                                const int       n,
                                const T*        alpha,
                                const T* const  x[],
                                const int       incx,
                                const T* const  y[],
                                const int       incy,
                                T* const        A[],
                                const int       lda,
                                const size_t    batch_count)
{

  constexpr size_t ts=16;

  const size_t m_max=((m+ts-1)/ts)*ts;
  const size_t n_max=((n+ts-1)/ts)*ts;

  return handle.submit([&](handler& cgh) {
      cgh.parallel_for(nd_range<3>{{batch_count,m_max,n_max},{1,ts,ts}},
          [=](nd_item<3> item) { // [[cl::intel_reqd_sub_group_size(32)]] {
          unsigned batch = item.get_global_id(0);
          unsigned x_g = item.get_global_id(1);
          unsigned y_g = item.get_global_id(2);
          if(x_g<m && y_g<n) 
          A[batch][x_g*lda + y_g] += alpha[batch]*x[batch][y_g]*y[batch][x_g];
          });
      });
}

template<typename T>
inline sycl::event ger_batch_strided(sycl::queue& handle,
                                const int       m,
                                const int       n,
                                const T*        alpha,
                                const T* const  x_b,
                                const int       incx,
                                const T* const  y_b,
                                const int       incy,
                                T* const        A_b,
                                const int       lda,
                                const size_t    batch_count)
{

  constexpr size_t ts=16;

  const size_t m_max=((m+ts-1)/ts)*ts;
  const size_t n_max=((n+ts-1)/ts)*ts;

  return handle.submit([&](handler& cgh) {
      cgh.parallel_for(nd_range<3>{{batch_count,m_max,n_max},{1,ts,ts}},
          [=](nd_item<3> item) { // [[cl::intel_reqd_sub_group_size(32)]] {
          unsigned batch = item.get_global_id(0);
          unsigned x_g = item.get_global_id(1);
          unsigned y_g = item.get_global_id(2);

          const T* restrict x=x_b+n*batch;
          const T* restrict y=y_b+m*batch;
          T* restrict A=A_b+m*lda*batch;
          if(x_g<m && y_g<n) 
          A[x_g*lda + y_g] += alpha[batch]*x[y_g]*y[x_g];
          });
      });
}
} // namespace 

} // namespace qmcplusplus
#endif // QMCPLUSPLUS_OMPBLAS_H
