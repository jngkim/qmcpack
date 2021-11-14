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
template<typename T, typename DT, typename index_t>
  inline std::complex<T> computeLogDet(const DT* restrict inv_mat, int n, int lda, 
      const index_t* restrict pivot)
  {
    std::complex<T> logdet{};
    for (size_t i = 0; i < n; i++)
      logdet += std::log(std::complex<T>((pivot[i] == i + 1) ? inv_mat[i*lda+i] : -inv_mat[i*lda+i]));
    return logdet;
  }

template<typename T, typename DT, typename Index_t>
  inline std::complex<T> computeLogDet(sycl::queue& aq, const DT* restrict a, 
      int n, int lda, const Index_t* restrict pivot)
  {
    constexpr size_t BS=128;
    std::complex<T> result{};
    {
      sycl::buffer<std::complex<T>,1> abuff(&result,{1});
      aq.submit([&](sycl::handler& cgh) {
          size_t n_max=((n+BS-1)/BS)*BS;
          sycl::global_ptr<const DT>  A{a};
          sycl::global_ptr<const Index_t>  Pivot{pivot};
          cgh.parallel_for(sycl::range<1>{n_max},
              sycl::reduction(abuff,cgh,{T{},T{}},std::plus<std::complex<T>>()), 
              [=](sycl::id<1> i, auto& sum)
              {
                  std::complex<T> val{};
                  if(i<n) val= std::log(std::complex<T>((Pivot[i] == i + 1) ? A[i*lda+i] : -A[i*lda+i]));
                  sum.combine(val);
              });
          });
    } 
    return  result;
  }

template<typename T, typename DT, typename Index_t>
  inline std::complex<T> computeLogDet_ND(sycl::queue& aq, const DT* restrict a, 
      int n, int lda, const Index_t* restrict pivot)
  {
    constexpr size_t BS=256;
    std::complex<T> result{};
    {
      sycl::buffer<std::complex<T>,1> abuff(&result,{1});
      aq.submit([&](sycl::handler& cgh) {
          size_t n_max=((n+BS-1)/BS)*BS;
          sycl::global_ptr<const DT>  A{a};
          sycl::global_ptr<const Index_t>  Pivot{pivot};
          cgh.parallel_for(sycl::nd_range<1>{{size_t(n_max)},{BS}}, 
              sycl::reduction(abuff,cgh,{T{},T{}},std::plus<std::complex<T>>()), 
              [=](sycl::nd_item<1> item, auto& sum)
              {
                  const unsigned i=item.get_global_id(0);
                  std::complex<T> val{};
                  if(i<n) val= std::log(std::complex<T>((Pivot[i] == i + 1) ? A[i*lda+i] : -A[i*lda+i]));
                  sum.combine(val);
              });
          });
    } 
    return  result;
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


  /** utilities for debugging */
  inline double inverse_gflops(size_t N, double t)
  {
    double dn=N;
    return 2.0e-9 * (4./3.)*dn*dn*dn/t;
  }

}
#endif
