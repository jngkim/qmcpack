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


#ifndef QMCPLUSPLUS_SYCL_ONEAPI_BLAS_H
#define QMCPLUSPLUS_SYCL_ONEAPI_BLAS_H

#include <complex>
#include <CL/sycl.hpp>
#include "mkl.h"
#include "oneapi/mkl/blas.hpp"

namespace qmcplusplus
{

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
                    const int          incy,
                    const std::vector<sycl::event> &events = {})
{
  return oneapi::mkl::blas::gemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y,incy,events);
}

template<typename T>
inline sycl::event gemv_batched(sycl::queue&   handle,
                    const oneapi::mkl::transpose trans,
                    const syclBLAS_int  m,
                    const syclBLAS_int  n,
                    const T*            alpha,
                    const T**           A,
                    const syclBLAS_int  lda,
                    const T**           X,
                    const syclBLAS_int  incx,
                    const T*            beta,
                    T**                 Y,
                    const syclBLAS_int  incy,
                    const syclBLAS_int  batch_count,
                    const std::vector<sycl::event> &events = {})
{
  //using group api: only one group
  return oneapi::mkl::blas::gemv_batch(handle, &trans, &m, &n, alpha, A, &lda,
      X,&incx, beta, Y, &incy,1,&batch_count, events);
}

/** special gemv_batched with multiple alpha values */
template<typename T>
inline int gemv_batched_alpha(sycl::queue&   handle,
                    const oneapi::mkl::transpose trans,
                    const syclBLAS_int  m,
                    const syclBLAS_int  n,
                    const T*            alpha,
                    const syclBLAS_int  nalphas, //extra argument (cheat)
                    const T**           a,
                    const syclBLAS_int  lda,
                    const T**           x,
                    const syclBLAS_int  incx,
                    const T             beta,
                    T**                 y,
                    const syclBLAS_int  incy,
                    const syclBLAS_int  batch_count,
                    const std::vector<sycl::event> &events = {})
{
  if(nalphas < batch_count) return 1;
  sycl::event::wait(events);
  std::vector<sycl::event> gemv_events(batch_count);
  for(int batch=0; batch<batch_count; batch++)
  {
    gemv_events[batch] = oneapi::mkl::blas::gemv(handle, trans, m, n,
                          alpha[batch], a[batch], lda, x[batch], incx, beta, y[batch],incy);
  }
  sycl::event::wait(gemv_events);
  return 0;
}

template<typename T>
inline sycl::event gemm_batched(sycl::queue&   handle,
                    const oneapi::mkl::transpose transA,
                    const oneapi::mkl::transpose transB,
                    const syclBLAS_int  m,
                    const syclBLAS_int  n,
                    const syclBLAS_int  k,
                    const T*            alpha,
                    const T**           A,
                    const syclBLAS_int  lda,
                    const T**           B,
                    const syclBLAS_int  ldb,
                    const T*            beta,
                    T**                 C,
                    const syclBLAS_int  ldc,
                    const syclBLAS_int  batch_count,
                    const std::vector<sycl::event> &events = {})
{
  //using group api: only one group
  return oneapi::mkl::blas::gemm_batch(handle, &transA, &transB, &m, &n, &k, alpha, A, &lda, B,&ldb, beta, C, &ldc,1,&batch_count, events);
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

  return handle.parallel_for(sycl::nd_range<2>{{m_max,n_max},{ts,ts}},
          [=](sycl::nd_item<2> item) { 
          unsigned x_g = item.get_global_id(0);
          unsigned y_g = item.get_global_id(1);
          if(x_g<m && y_g<n) 
          A[x_g*lda + y_g] += alpha*x[y_g]*y[x_g];
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

  return handle.parallel_for(
      sycl::nd_range<3>{{batch_count,m_max,n_max},{1,ts,ts}},
          [=](sycl::nd_item<3> item) { // [[cl::intel_reqd_sub_group_size(32)]] {
          unsigned batch = item.get_global_id(0);
          unsigned x_g = item.get_global_id(1);
          unsigned y_g = item.get_global_id(2);
          if(x_g<m && y_g<n) 
          A[batch][x_g*lda + y_g] += alpha[batch]*x[batch][y_g]*y[batch][x_g];
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

  return handle.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<3>{{batch_count,m_max,n_max},{1,ts,ts}},
          [=](sycl::nd_item<3> item) { // [[cl::intel_reqd_sub_group_size(32)]] {
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

template<typename T1, typename T2>
inline sycl::event transpose(sycl::queue& q, 
    const T1* restrict in, int m, int lda, T2* restrict out, int n, int ldb, size_t tile_size=16)
{
  const size_t m_max=((m+tile_size-1)/tile_size)*tile_size;
  const size_t n_max=((n+tile_size-1)/tile_size)*tile_size;

  return q.submit([&](sycl::handler& cgh) {

      sycl::accessor<T2, 2, sycl::access::mode::write, sycl::access::target::local> 
      tile(sycl::range<2>(tile_size,tile_size+1), cgh);

      cgh.parallel_for(sycl::nd_range<2>{{m_max,n_max},{tile_size,tile_size}},
          [=](sycl::nd_item<2> item) { 
          unsigned x = item.get_global_id(1);
          unsigned y = item.get_global_id(0);
          unsigned xth=item.get_local_id(1);
          unsigned yth=item.get_local_id(0);

          tile[yth][xth] = in[(y)*lda + x];
          item.barrier(sycl::access::fence_space::local_space);

          x = item.get_group(0)*tile_size + xth;
          y = item.get_group(1)*tile_size + yth;
          if(x<m && y<n)   out[(y)*ldb + x] = tile[xth][yth]; 
          });
      });
  }

template<typename T1, typename T2>
inline sycl::event transpose_batched_2D(sycl::queue& q, 
    const T1* restrict in_b, int m, int lda, T2* restrict out_b, int n, int ldb, 
    int batch_size, size_t tile_size=16)
{
  const size_t m_max=((m+tile_size-1)/tile_size)*tile_size;
  const size_t n_max=((n+tile_size-1)/tile_size)*tile_size;

  return q.submit([&](sycl::handler& cgh) {

      sycl::accessor<T2, 2, sycl::access::mode::write, sycl::access::target::local> 
      tile(sycl::range<2>(tile_size,tile_size+1), cgh);

      cgh.parallel_for(sycl::nd_range<2>{{m_max,n_max},{tile_size,tile_size}},
          [=](sycl::nd_item<2> item) { // [[cl::intel_reqd_sub_group_size(32)]] {
          const unsigned xth=item.get_local_id(1);
          const unsigned yth=item.get_local_id(0);
          const unsigned x_in = item.get_global_id(1);
          const unsigned y_in = item.get_global_id(0);
          const unsigned x_out = item.get_group(0)*tile_size + xth;
          const unsigned y_out = item.get_group(1)*tile_size + yth;

          for(int batch=0; batch<batch_size; ++batch)
          {
          const T1* restrict in=in_b+m*lda*batch;

          tile[yth][xth] = in[(y_in)*lda + x_in];
          item.barrier(sycl::access::fence_space::local_space);

          T2* restrict out=out_b+n*ldb*batch;
          if(x_in<m && y_out<n)   out[(y_out)*ldb + x_out] = tile[xth][yth]; 
          }
          });
      });
  }

template<typename T1, typename T2>
inline sycl::event transpose_batched(sycl::queue& q, 
    const T1** restrict in_b, int m, int lda, T2** restrict out_b, int n, int ldb, 
    int batch_size, size_t tile_size=16)
{
  const size_t m_max=((m+tile_size-1)/tile_size)*tile_size;
  const size_t n_max=((n+tile_size-1)/tile_size)*tile_size;

  return q.submit([&](sycl::handler& cgh) {

      sycl::accessor<T2, 2, sycl::access::mode::write, sycl::access::target::local> 
      tile(sycl::range<2>(tile_size,tile_size+1), cgh);

      cgh.parallel_for(sycl::nd_range<3>{{size_t(batch_size),m_max,n_max},{1,tile_size,tile_size}},
          [=](sycl::nd_item<3> item) { // [[cl::intel_reqd_sub_group_size(32)]] {
          const unsigned xth=item.get_local_id(2);
          const unsigned yth=item.get_local_id(1);
          const unsigned x_in = item.get_global_id(2);
          const unsigned y_in = item.get_global_id(1);
          const unsigned x_out = item.get_group(1)*tile_size + xth;
          const unsigned y_out = item.get_group(2)*tile_size + yth;

          const unsigned batch=item.get_global_id(0);
          const T1* restrict in=in_b[batch];

          tile[yth][xth] = in[(y_in)*lda + x_in];
          item.barrier(sycl::access::fence_space::local_space);

          T2* restrict out=out_b[batch];
          if(x_in<m && y_out<n)   out[(y_out)*ldb + x_out] = tile[xth][yth]; 
          });
      });
  }

template<typename T1, typename T2>
inline sycl::event transpose_batched_strided(sycl::queue& q, 
    const T1* restrict in_b, int m, int lda, T2* restrict out_b, int n, int ldb, 
    int batch_size, size_t tile_size=16)
{
  const size_t m_max=((m+tile_size-1)/tile_size)*tile_size;
  const size_t n_max=((n+tile_size-1)/tile_size)*tile_size;

  return q.submit([&](sycl::handler& cgh) {

      sycl::accessor<T2, 2, sycl::access::mode::write, sycl::access::target::local> 
      tile(sycl::range<2>(tile_size,tile_size+1), cgh);

      cgh.parallel_for(sycl::nd_range<3>{{size_t(batch_size),m_max,n_max},{1,tile_size,tile_size}},
          [=](sycl::nd_item<3> item) { // [[cl::intel_reqd_sub_group_size(32)]] {
          const unsigned xth=item.get_local_id(2);
          const unsigned yth=item.get_local_id(1);
          const unsigned x_in = item.get_global_id(2);
          const unsigned y_in = item.get_global_id(1);
          const unsigned x_out = item.get_group(1)*tile_size + xth;
          const unsigned y_out = item.get_group(2)*tile_size + yth;

          const unsigned batch=item.get_global_id(0);
          const T1* restrict in=in_b+m*lda*batch;

          tile[yth][xth] = in[(y_in)*lda + x_in];
          item.barrier(sycl::access::fence_space::local_space);

          T2* restrict out=out_b+n*ldb*batch;
          if(x_in<m && y_out<n)   out[(y_out)*ldb + x_out] = tile[xth][yth]; 
          });
      });
  }

  template <typename T1, typename T2>
    inline sycl::event
    copy_n(sycl::queue &aq, const T1* restrict VA, size_t array_size, T2* restrict VC)
    {
      return aq.parallel_for({array_size}, [=](sycl::id<1> id) {
          VC[id] = static_cast<T2>(VA[id]);
          });
    }


  template <typename T1, typename T2>
    inline sycl::event
    copy(sycl::queue &aq, unsigned m, unsigned n,
        const T1* restrict matA, int lda, T2* restrict matB, int ldb,
        const size_t COLBS=16)
    { //NO NEED 2D
      const size_t m_max=((m+COLBS-1)/COLBS)*COLBS;
      const size_t n_max=((n+COLBS-1)/COLBS)*COLBS;
      return aq.parallel_for(
          sycl::nd_range<2>{{m_max,n_max},{COLBS,COLBS}},
          [=](sycl::nd_item<2> item) { // [[cl::intel_reqd_sub_group_size(32)]] {
          unsigned i = item.get_global_id(0);
          unsigned j = item.get_global_id(1);
          if(i<n && j<m)
            matB[j+i*ldb]=matA[j+i*lda];
          });
    }
} // namespace 

} // namespace qmcplusplus
#endif // QMCPLUSPLUS_OMPBLAS_H
