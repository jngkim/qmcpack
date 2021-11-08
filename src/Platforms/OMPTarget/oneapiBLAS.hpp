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

namespace qmcplusplus
{
/** Implement selected batched and non-batched BLAS2 calls using OpenMP offload for different data types S/C/D/Z
 * 1) column major like the BLAS fortran API
 * 2) all the functions are synchronous, expected to be changed to asynchronous in the future.
 * 3) all the poompBLAS_inter arguments are expected as device poompBLAS_inters.
 * 4) in batched APIs, alpha and beta are **not** scalars but poompBLAS_inters to array of batch size.
 */
namespace ompBLAS
{

typedef std::int64_t ompBLAS_int;
typedef int ompBLAS_status;
typedef int ompBLAS_handle;

template<typename T>
inline ompBLAS_status gemv(ompBLAS_handle&    handle,
                    const char         trans,
                    const ompBLAS_int          m,
                    const ompBLAS_int          n,
                    const T        alpha,
                    const T* const A,
                    const ompBLAS_int          lda,
                    const T* const x,
                    const ompBLAS_int          incx,
                    const T        beta,
                    T* const       y,
                    const ompBLAS_int          incy)
{
  const auto transA = oneapi::mkl::transpose::trans;
  sycl::queue *main_queue=get_default_queue();
  oneapi::mkl::blas::gemv(*main_queue, transA, m, n, alpha, A, lda, x, incx, beta, y,incy).wait();
  return 0;
}

template<typename T>
ompBLAS_status gemv_batched(ompBLAS_handle&    handle,
                            const char         trans,
                            const ompBLAS_int          m,
                            const ompBLAS_int          n,
                            const T*       alpha,
                            const T**      A,
                            const ompBLAS_int          lda,
                            const T**      x,
                            const ompBLAS_int          incx,
                            const T*       beta,
                            T**            y,
                            const ompBLAS_int          incy,
                            const ompBLAS_int          batch_count)
{
  const oneapi::mkl::transpose tA= oneapi::mkl::transpose::trans;
  //const T alpha_=T(1.0);
  //const T beta_=T(0.0);
  sycl::queue *main_queue=get_default_queue();
  oneapi::mkl::blas::gemv_batch(*main_queue, &tA, &m, &n, alpha, A, &lda,x,&incx, beta, y, &incy,1,&batch_count).wait();
  return 0;
}

template<typename T>
ompBLAS_status ger(ompBLAS_handle& handle,
                        const ompBLAS_int       m,
                        const ompBLAS_int       n,
                        const T         alpha,
                        const T* const  x,
                        const ompBLAS_int       incx,
                        const T* const  y,
                        const ompBLAS_int       incy,
                        T* const        A,
                        const ompBLAS_int       lda)
{
  if (incx !=1 || incy != 1)
    throw std::runtime_error("incx !=1 or incy != 1 are not implemented in ompBLAS::ger_impl!");

  //BLAS::ger(m, n, alpha, x, incx, y, incy, A, lda);
  PRAGMA_OFFLOAD("omp target teams distribute parallel for collapse(2) is_device_ptr(A, x, y)")
  for(size_t i = 0; i < n; i++)
    for(size_t j = 0; j < m; j++)
      A[i * lda + j] += alpha * x[j] * y[i];
  return 0;
}

template<typename T>
ompBLAS_status ger_batched(ompBLAS_handle& handle,
                                const ompBLAS_int       m,
                                const ompBLAS_int       n,
                                const T*        alpha,
                                const T* const  x[],
                                const ompBLAS_int       incx,
                                const T* const  y[],
                                const ompBLAS_int       incy,
                                T* const        A[],
                                const ompBLAS_int       lda,
                                const ompBLAS_int       batch_count)
{
  if (batch_count == 0) return 0;

  if (incx !=1 || incy != 1)
    throw std::runtime_error("incx !=1 or incy != 1 are not implemented in ompBLAS::ger_batched_impl!");

  PRAGMA_OFFLOAD("omp target teams distribute parallel for collapse(3) is_device_ptr(A, x, y, alpha)")
  for(size_t ib = 0; ib < batch_count; ib++)
    for(size_t i = 0; i < n; i++)
      for(size_t j = 0; j < m; j++)
        A[ib][i * lda + j] += alpha[ib] * x[ib][j] * y[ib][i];
  return 0;
}

} // namespace ompBLAS

} // namespace qmcplusplus
#endif // QMCPLUSPLUS_OMPBLAS_H
