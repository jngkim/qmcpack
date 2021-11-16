//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_SYCL_DETERMINANT_HELPER_H
#define QMCPLUSPLUS_SYCL_DETERMINANT_HELPER_H

namespace qmcplusplus
{
template<typename T, typename TMAT, typename index_t>
  inline std::complex<T> computeLogDet(int n, 
                                       int lda, 
                                       const TMAT* restrict inv_mat, 
                                       const index_t* restrict pivot)
  {
    std::complex<T> logdet{};
    for (size_t i = 0; i < n; i++)
      logdet += std::log(std::complex<T>((pivot[i] == i + 1) ? inv_mat[i*lda+i] : -inv_mat[i*lda+i]));
    return logdet;
  }

/** compute logdet of a single array using SYCL::reduction 
 */
template<typename T, typename TMAT, typename Index_t>
inline std::complex<T> computeLogDet(sycl::queue& aq, 
                                     int n, 
                                     int lda, 
                                     const TMAT* restrict a, 
                                     const Index_t* restrict pivot)
  {
    constexpr size_t COLBS=32;
    constexpr std::complex<T> log_One{T(1),T(0)};
    constexpr std::complex<T> log_mOne{T(-1),T(0)};

    std::complex<T> result{};
    {
      sycl::buffer<std::complex<T>,1> abuff(&result,{1});
      aq.submit([&](sycl::handler& cgh) {
          size_t n_max=((n+COLBS-1)/COLBS)*COLBS;
          sycl::global_ptr<const TMAT>  A{a};
          sycl::global_ptr<const Index_t>  Pivot{pivot};
          cgh.parallel_for(sycl::range<1>{n_max},
              sycl::reduction(abuff,cgh,{T{},T{}},std::plus<std::complex<T>>()), 
              [=](sycl::id<1> i, auto& sum)
              {
                  std::complex<T> val{};
                  //if(i<n) val= std::log(std::complex<T>((Pivot[i] == i + 1) ? A[i*lda+i] : -A[i*lda+i]));
                  if(i<n) 
                  val = (Pivot[i] == i + 1) ? 
                    std::log(std::complex<T>(A[i*lda+i])) : std::log(std::complex<T>(-A[i*lda+i]));
                  sum.combine(val);
              });
          });
    } //synchronous
    return  result;
  }

template<typename T, typename TMAT, typename Index_t>
  inline std::complex<T> computeLogDetNDR(sycl::queue& aq, 
      int n, int lda, 
      const TMAT* restrict a, 
      const Index_t* restrict pivot,
      const size_t COLBS=256)
  {
    std::complex<T> result{};
    {
      sycl::buffer<std::complex<T>,1> abuff(&result,{1});
      aq.submit([&](sycl::handler& cgh) {
          size_t n_max=((n+COLBS-1)/COLBS)*COLBS;
          sycl::global_ptr<const TMAT>  A{a};
          sycl::global_ptr<const Index_t>  Pivot{pivot};
          cgh.parallel_for(sycl::nd_range<1>{{size_t(n_max)},{COLBS}}, 
              sycl::reduction(abuff,cgh,{T{},T{}},std::plus<std::complex<T>>()), 
              [=](sycl::nd_item<1> item, auto& sum)
              {
                  const unsigned i=item.get_global_id(0);
                  std::complex<T> val{};
                  //if(i<n) val= std::log(std::complex<T>((Pivot[i] == i + 1) ? A[i*lda+i] : -A[i*lda+i]));
                  if(i<n) 
                  val = (Pivot[i] == i + 1) ? 
                    std::log(std::complex<T>(A[i*lda+i])) : std::log(std::complex<T>(-A[i*lda+i]));
                  sum.combine(val);
              });
          });
    } 
    return  result;
  }

template<typename TMAT, typename T, typename Index_t>
  inline sycl::event computeLogDet_batched(sycl::queue& aq, 
                                   int n, 
                                   int lda, 
                                   const TMAT* restrict mat_lus, 
                                   const Index_t* restrict pivots,
                                   std::complex<T>* restrict logdets,
                                   const int batch_size,
                                   const size_t COLBS=128)
  {
    return aq.submit([&](sycl::handler& cgh) {

        sycl::accessor<std::complex<T>, 1, sycl::access::mode::write, sycl::access::target::local> 
        logdet_vals(sycl::range<1>{COLBS}, cgh);

        cgh.parallel_for(sycl::nd_range<1>{{batch_size*COLBS},{COLBS}}, 
            [=](sycl::nd_item<1> item) {
            const unsigned iw  = item.get_group(0);
            const unsigned tid = item.get_local_id(0);
            const Index_t* restrict pv_iw = pivots+iw*lda;
            const TMAT* restrict    lu_iw = mat_lus+iw*n*lda;

            std::complex<T> val{};
            const unsigned  num_col_blocks = (n + COLBS - 1) / COLBS;
            for (unsigned block_num = 0; block_num < num_col_blocks; block_num++)
            {
              const unsigned i = tid + block_num * COLBS;
              if(i<n)
                val += (pv_iw[i] == i + 1) ? 
                  std::log(std::complex<T>(lu_iw[i*lda+i])) : std::log(std::complex<T>(-lu_iw[i*lda+i]));
            }

            logdet_vals[tid]=val;

            for (unsigned iend = COLBS / 2; iend > 0; iend /= 2)
            {
              item.barrier(sycl::access::fence_space::local_space);
              if (tid < iend)
              {
                 logdet_vals[tid] += logdet_vals[tid + iend];
              }
            }
            if (tid == 0)
              logdets[iw] = logdet_vals[0];
            });
    });
  }

template<typename TMAT, typename T, typename Index_t>
  inline sycl::event computeLogDetGroup(sycl::queue& aq, 
                                   int n, 
                                   int lda, 
                                   const TMAT* restrict mat_lus, 
                                   const Index_t* restrict pivots,
                                   std::complex<T>* restrict logdets,
                                   const size_t batch_size,
                                   const size_t COLBS=128)
  {
    return aq.parallel_for(sycl::nd_range<1>{{batch_size*COLBS},{COLBS}}, 
            [=](sycl::nd_item<1> item) {
            const unsigned iw  = item.get_group(0);
            const unsigned tid = item.get_local_id(0);
            const Index_t* restrict pv_iw = pivots+iw*lda;
            const TMAT* restrict    lu_iw = mat_lus+iw*n*lda;

            std::complex<T> val{};
            const unsigned  num_col_blocks = (n + COLBS - 1) / COLBS;
            for (unsigned block_num = 0; block_num < num_col_blocks; block_num++)
            {
              const unsigned i = tid + block_num * COLBS;
              if(i<n)
                val += (pv_iw[i] == i + 1) ? 
                  std::log(std::complex<T>(lu_iw[i*lda+i])) : std::log(std::complex<T>(-lu_iw[i*lda+i]));
            }

            val = sycl::reduce_over_group(item.get_group(),val,{T{},T{}},std::plus<std::complex<T>>());

            if(iw==0)
              logdets[iw] = val;
            });
  }

  template<typename T, typename Index_t>
  inline sycl::event 
  applyW_stageV(sycl::queue& aq, T* restrict UV ,const size_t norb, 
      const Index_t* restrict delay_list, const size_t delay_count)
  {
    constexpr T mone(-1);

    if(delay_count < 16)
      return aq.submit([&](sycl::handler& cgh) {
          cgh.single_task([=](){
              for(int i=0; i<delay_count; ++i)
              UV[delay_list[i]*delay_count+i] += mone;
              }); 
          });
    else
      return aq.parallel_for(sycl::nd_range<1>{{delay_count},{delay_count}}, 
          [=](sycl::nd_item<1> item) {
          const unsigned i=item.get_global_id(0);
          UV[delay_list[i]*delay_count+i] += mone;
          });
  }

template<typename T, size_t COLBS>
inline void copyAinvRow_saveGL(sycl::queue& aq,
                               const int rowchanged,
                               const int n,
                               const T* const Ainv[],
                               const int lda,
                               T* const temp[],
                               T* const rcopy[],
                               const T* const dphi_in[],
                               const T* const d2phi_in[],
                               T* const dphi_out[],
                               T* const d2phi_out[],
                               const size_t batch_count)
{
  return aq.parallel_for(sycl::nd_range<1>{{batch_count*COLBS},{COLBS}}, 
      [=](sycl::nd_item<1> item) {
      const int iw                      = item.get_group(0);
      const T* __restrict__ Ainv_iw     = Ainv[iw];
      T* __restrict__ temp_iw           = temp[iw];
      T* __restrict__ rcopy_iw          = rcopy[iw];
      const T* __restrict__ dphi_in_iw  = dphi_in[iw];
      const T* __restrict__ d2phi_in_iw = d2phi_in[iw];
      T* __restrict__ dphi_out_iw       = dphi_out[iw];
      T* __restrict__ d2phi_out_iw      = d2phi_out[iw];

      const unsigned tid = item.get_local_id(0);
      if (tid == 0)
        temp_iw[rowchanged] = -temp_iw[rowchanged];

      const unsigned num_col_blocks = (n + COLBS - 1) / COLBS;
      for (unsigned ib = 0; ib < num_col_blocks; ib++)
      {
        const unsigned col_id = ib * COLBS + tid; 
        if (col_id < n)
        {
          rcopy_iw[col_id] = Ainv_iw[rowchanged * lda + col_id];

          // the following copying data on the device is not part of SM-1
          // it is intended to copy dphiV and d2phiV from temporary to final without a separate kernel.
          dphi_out_iw[col_id * 3]     = dphi_in_iw[col_id * 3];
          dphi_out_iw[col_id * 3 + 1] = dphi_in_iw[col_id * 3 + 1];
          dphi_out_iw[col_id * 3 + 2] = dphi_in_iw[col_id * 3 + 2];
          d2phi_out_iw[col_id]        = d2phi_in_iw[col_id];
        }
      }
      });
}


  /** utilities for debugging */
  inline double inverse_gflops(size_t N, double t)
  {
    double dn=N;
    return 2.0e-9 * (4./3.)*dn*dn*dn/t;
  }

}
#endif
