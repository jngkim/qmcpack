//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel corporation
//
// File created by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_DIRAC_MATRIX_COMPUTE_SYCL_MKL_H
#define QMCPLUSPLUS_DIRAC_MATRIX_COMPUTE_SYCL_MKL_H

#include "OhmmsPETE/OhmmsMatrix.h"
#include "OMPTarget/OMPallocator.hpp"
#include "Platforms/PinnedAllocator.h"
#include "type_traits/complex_help.hpp"
#include "type_traits/template_types.hpp"
#include "Message/OpenMP.h"
#include "CPU/SIMD/simd.hpp"
#include "SYCL/SYCLruntime.hpp"
#include "SYCL/SYCLallocator.hpp"
#include "SYCL/syclBLAS.hpp"
#include "QMCWaveFunctions/detail/SYCL/sycl_determinant_helper.hpp"
#include "ResourceCollection.h"
#include "oneapi/mkl/lapack.hpp"
#include "mkl.h"

namespace qmcplusplus
{

//use MKL API directly
namespace syclSolver=oneapi::mkl::lapack;

/** class to compute matrix inversion and the log value of determinant
 *  of a batch of DiracMatrixes.
 *
 *  @tparam VALUE_FP the datatype used in the actual computation of the matrix
 *  
 *  There is one per crowd not one per MatrixUpdateEngine.
 *  this puts ownership of the scratch resources in a sensible place.
 *
 *  This is compatible with DiracMatrixComputeOMPTarget and can be used both on CPU
 *  and GPU when the resrouce management is properly handled.
 */
template<typename VALUE_FP>
class DiracMatrixComputeSYCL : public Resource
{
public:
  using FullPrecReal = RealAlias<VALUE_FP>;
  using LogValue     = std::complex<FullPrecReal>;

  template<typename T>
  using OffloadPinnedAllocator = OMPallocator<T, PinnedAlignedAllocator<T>>;
  template<typename T>
  using OffloadPinnedMatrix = Matrix<T, OffloadPinnedAllocator<T>>;
  template<typename T>
  using OffloadPinnedVector = Vector<T, OffloadPinnedAllocator<T>>;

  //sycl::queue managed by MatrixDelayedUpdateSYCL
  using HandleResource = sycl::queue;

private:

  template<typename T>
  using DeviceVector = Vector<T, SYCLAllocator<T>>;

  sycl::queue *m_queue=nullptr;;
  std::int64_t getrf_ws=0;
  std::int64_t getri_ws=0;
  std::int64_t lwork_=0;

  std::vector<sycl::event> batch_events;

  DeviceVector<VALUE_FP> psiM_fp_;
  DeviceVector<VALUE_FP> m_work_;  
  DeviceVector<std::int64_t> pivots_;

  /** reset internal work space.
   */
  inline void reset(const int n, const int lda, const int batch_size)
  {
    //possible to use in_order_queue 
    if(m_queue==nullptr) m_queue=get_default_queue();

    const int nw = batch_size;
    getrf_ws=syclSolver::getrf_scratchpad_size<VALUE_FP>(*m_queue,n,n,lda);
    getri_ws=syclSolver::getri_scratchpad_size<VALUE_FP>(*m_queue,n,lda);
    lwork_=std::max(getrf_ws,getri_ws);

    psiM_fp_.resize(nw*n*lda);
    m_work_.resize(nw*lwork_);
    pivots_.resize(nw*lda);

    batch_events.resize(nw);
  }

public:
  DiracMatrixComputeSYCL() : Resource("DiracMatrixComputeOMPTarget") {}

  Resource* makeClone() const override { return new DiracMatrixComputeSYCL(*this); }

  /** compute the inverse of the transpose of matrix A and its determinant value in log
   * when VALUE_FP and TMAT are the same
   * @tparam TMAT matrix value type
   * @tparam TREAL real type
   * \param [in]    resource          compute resource
   * \param [in]    a_mat             matrix to be inverted
   * \param [out]   inv_a_mat         the inverted matrix
   * \param [out]   log_value         breaks compatibility of MatrixUpdateOmpTarget with
   *                                  DiracMatrixComputeCUDA but is fine for OMPTarget        
   */
  template<typename TMAT>
  inline std::enable_if_t<std::is_same<VALUE_FP, TMAT>::value> invert_transpose(HandleResource& resource,
                                                                                const OffloadPinnedMatrix<TMAT>& a_mat,
                                                                                OffloadPinnedMatrix<TMAT>& inv_a_mat,
                                                                                LogValue& log_value)
  {
    const size_t n   = a_mat.rows();
    const size_t lda = a_mat.cols();
    const size_t ldb = inv_a_mat.cols();

    if(pivots_.size() < lda) reset(n,lda,1);

    //all blocking: m_queue can be an inorder queue per object
    syclBLAS::transpose(*m_queue,a_mat.device_data(),n,lda,inv_a_mat.device_data(),n,ldb).wait();
    syclSolver::getrf(*m_queue,n,n,inv_a_mat.device_data(),lda, pivots_.data(), m_work_.data(), getrf_ws).wait();
    log_value=computeLogDet<FullPrecReal>(*m_queue, n, lda, inv_a_mat.device_data(), pivots_.data());
    syclSolver::getri(*m_queue,n,inv_a_mat.device_data(),lda, pivots_.data(), m_work_.data(), getri_ws).wait();

    inv_a_mat.updateFrom();
  }

  /** compute the inverse of the transpose of matrix A and its determinant value in log
   * when VALUE_FP and TMAT are the different
   * @tparam TMAT matrix value type
   * @tparam TREAL real type
   */
  template<typename TMAT>
  inline std::enable_if_t<!std::is_same<VALUE_FP, TMAT>::value> invert_transpose(HandleResource& resource,
                                                                                 const OffloadPinnedMatrix<TMAT>& a_mat,
                                                                                 OffloadPinnedMatrix<TMAT>& inv_a_mat,
                                                                                 LogValue& log_value)
  {
    const int n   = a_mat.rows();
    const int lda = a_mat.cols();
    const int ldb = inv_a_mat.cols();

    if(pivots_.size() < lda) reset(n,lda,1);

    syclBLAS::transpose(*m_queue,a_mat.device_data(),n,lda,psiM_fp_.data(),n,ldb).wait();
    syclSolver::getrf(*m_queue,n,n,psiM_fp_.data().device_data(),lda, pivots_.data(), m_work_.data(), getrf_ws).wait();
    log_value=computeLogDet<FullPrecReal>(*m_queue,inv_a_mat.device_data(), n, lda, pivots_.data());
    syclSolver::getri(*m_queue,n,psiM_fp_.data(),lda, pivots_.data(), m_work_.data(), getri_ws).wait();
    syclBLAS::copy(*m_queue, n, n, psiM_fp_.data(), lda, inv_a_mat.device_data(),ldb).wait();

    inv_a_mat.updateFrom();
  }

  /** This covers both mixed and Full precision case.
   *  
   *  \todo measure if using the a_mats without a copy to contiguous vector is better.
   */
  template<typename TMAT>
  inline void mw_invertTranspose(HandleResource& resource,
                                 const RefVector<const OffloadPinnedMatrix<TMAT>>& a_mats,
                                 const RefVector<OffloadPinnedMatrix<TMAT>>& inv_a_mats,
                                 OffloadPinnedVector<LogValue>& log_values)
  {
#if 0
    //invert_transpose
    for (int iw = 0; iw < a_mats.size(); iw++)
    {
      invert_transpose(resource,a_mats[iw].get(),inv_a_mats[iw].get(),log_values[iw]);
    }
#else
    const int nw  = a_mats.size();
    const int n   = a_mats[0].get().rows();
    const int lda = a_mats[0].get().cols();
    const int ldb = inv_a_mats[0].get().cols();
    const int nsqr{n * lda};

    if(pivots_.size()<nw*lda) reset(n,lda,nw);

    for (int iw = 0; iw < nw; ++iw)
      batch_events[iw] = syclBLAS::transpose(*m_queue, a_mats[iw].get().device_data(),n,lda, psiM_fp_.data()+iw*nsqr,n,ldb); 

#if MKL_BATCHED_INVERSE
    syclSolver::getrf_batch(*m_queue,n,n,psiM_fp_.data(),lda,nsqr,
        pivots_data(), lda, batch_count, m_work_.data(),getrf_ws*nw,batch_events).wait();
#else
    for (int iw = 0; iw < nw; ++iw)
      batch_events[iw]= syclSolver::getrf(*m_queue,n,n,psiM_fp_.data()+iw*nsqr,lda, 
          pivots_.data()+iw*lda, m_work_.data()+iw*getrf_ws, getrf_ws, {batch_events[iw]});
    sycl::event::wait(batch_events);
#endif

    computeLogDet_batched(*m_queue,n,lda,
        psiM_fp_.data(),pivots_.data(),log_values.device_data(),nw).wait();

#if MKL_BATCHED_INVERSE
    syclSolver::getrf_batch(*m_queue,n,psiM_fp_.data(),lda,nsqr,
        pivots_data(), lda, batch_count, m_work_.data(),getri_ws*nw).wait();
#else
    for (int iw = 0; iw < nw; ++iw)
      batch_events[iw]= syclSolver::getri(*m_queue,n,psiM_fp_.data()+iw*nsqr,lda, 
          pivots_.data()+lda*iw, m_work_.data()+iw*getri_ws, getrf_ws);
    sycl::event::wait(batch_events);
#endif

    for (int iw = 0; iw < nw; ++iw)
    {
      syclBLAS::copy_n(*m_queue, psiM_fp_.data() + nsqr * iw, n*lda, 
          inv_a_mats[iw].get().device_data()).wait();
      inv_a_mats[iw].get().updateFrom(); 
    }
    log_values.updateFrom();
#endif
  }
};
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_DIRAC_MATRIX_COMPUTE_SCYL_H
