//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_MATRIX_DELAYED_UPDATE_SYCL_H
#define QMCPLUSPLUS_MATRIX_DELAYED_UPDATE_SYCL_H

#include "OhmmsPETE/OhmmsVector.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "DualAllocatorAliases.hpp"
#include "QMCWaveFunctions/Fermion/DiracMatrix.h"
#include "Platforms/OMPTarget/ompBLAS.hpp"
#include "SYCL/SYCLruntime.hpp"
#include "SYCL/SYCLallocator.hpp"
#include "SYCL/syclBLAS.hpp"
#include "QMCWaveFunctions/detail/SYCL/sycl_LU.hpp"
#include "DualAllocatorAliases.hpp"
#include "DiracMatrixComputeSYCL.hpp"
#include "ResourceCollection.h"
#include "WaveFunctionTypes.hpp"

namespace qmcplusplus
{

//use MKL API directly for GEMM
namespace mklBLAS=oneapi::mkl::blas;

/** implements dirac matrix delayed update using OpenMP offload and SYCL.
 * It is used as DET_ENGINE in DiracDeterminantBatched.
 * This is a 1 per walker class
 *
 * @tparam T base precision for most computation
 * @tparam T_FP high precision for matrix inversion, T_FP >= T
 */
template<typename VALUE, typename VALUE_FP>
class MatrixDelayedUpdateSYCL
{
public:
  using WFT           = WaveFunctionTypes<VALUE, VALUE_FP>;
  using Value         = typename WFT::Value;
  using FullPrecValue = typename WFT::FullPrecValue;
  using LogValue      = typename WFT::LogValue;
  using This_t        = MatrixDelayedUpdateSYCL<VALUE, VALUE_FP>;
  using DetInverter   = DiracMatrixComputeSYCL<FullPrecValue>;

  //use the same allocator as DetInverter,
  template<typename DT>
  using PinnedDualAllocator = DetInvert::OffloadPinnedAllocator<DT>;
  template<typename DT>
  using DualVector = Vector<DT, PinnedDualAllocator<DT>>;
  template<typename DT>
  using DualMatrix = Matrix<DT, PinnedDualAllocator<DT>>;

  //Use m_queue->memcpy to manage asynchronous copies
  template<typename DT>
  using DevicelAllocator = SYCLAllocator<DT>;
  template<typename DT>
  using DeviceVector = Vector<DT, SYCLAllocator<DT>>;

  struct MatrixDelayedUpdateSYCLMultiWalkerMem : public Resource
  {
    // multi walker of grads for transfer needs.
    DualMatrix<Value> grads_value_v;
    //potentially better to use SYCLHostAllocator
    // mw_updateRow pointer buffer
    DualVector<Value*> updateRow_buffer_H2D;
    // mw_prepareInvRow pointer buffer
    DualVector<Value*> prepare_inv_row_buffer_H2D;
    // mw_accept_rejectRow pointer buffer
    DualVector<Value*> accept_rejectRow_buffer_H2D;
    // mw_updateInv pointer buffer
    DualVector<Value*> updateInv_buffer_H2D;
    // mw_evalGrad pointer buffer
    DualVector<Value*> evalGrad_buffer_H2D;
    /// scratch space for rank-1 update
    DeviceVector<Value> mw_temp;
    // scratch space for keeping one row of Ainv
    DeviceVector<Value> mw_rcopy;
    // ratios for ger
    Vector<Value, SYCLHostAllocator<Value>> mw_ratio;

    MatrixDelayedUpdateSYCLMultiWalkerMem() : Resource("MatrixDelayedUpdateSYCLMultiWalkerMem") {}

    MatrixDelayedUpdateSYCLMultiWalkerMem(const MatrixDelayedUpdateSYCLMultiWalkerMem&)
        : MatrixDelayedUpdateSYCLMultiWalkerMem()
    {}

    Resource* makeClone() const override { return new MatrixDelayedUpdateSYCLMultiWalkerMem(*this); }
  };

  const DualMatrix<Value>& get_psiMinv() const { return psiMinv_; }
  DualMatrix<Value>& get_ref_psiMinv() { return psiMinv_; }

private:
  /// legacy single walker matrix inversion engine
  DiracMatrixComputeSYCL<FullPrecValue> detEng;
  /* inverse transpose of psiM(j,i) \f$= \psi_j({\bf r}_i)\f$
   * Only NumOrbitals x NumOrbitals subblock has meaningful data
   * The number of rows is equal to NumOrbitals
   * The number of columns in each row is padded to a multiple of QMC_SIMD_ALIGNMENT
   */
  DualMatrix<Value> psiMinv_;
  /// scratch space for rank-1 update
  DeviceVector<Value> temp;
  /// row of up-to-date Ainv
  DualVector<Value> invRow;
  /** row id correspond to the up-to-date invRow. [0 norb), invRow is ready; -1, invRow is not valid.
   *  This id is set after calling getInvRow indicating invRow has been prepared for the invRow_id row
   *  ratioGrad checks if invRow_id is consistent. If not, invRow needs to be recomputed.
   *  acceptMove and completeUpdates mark invRow invalid by setting invRow_id to -1
   */
  int invRow_id;
  // scratch space for keeping one row of Ainv
  DeviceVector<Value> rcopy;

  /// orbital values of delayed electrons
  DeviceMatrix<Value> U_gpu;
  /// rows of Ainv corresponding to delayed electrons
  DeviceMatrix<Value> V_gpu;
  /// Matrix inverse of B, at maximum KxK
  DeviceMatrix<Value> Binv_gpu;
  /// scratch space, used during inverse update
  DeviceMatrix<Value> tempMat_gpu;
  /// new column of B
  DeviceVector<Value> p_gpu;
  /// list of delayed electrons
  //DeviceVector<int> delay_list_gpu; //this can be pinned
  Vector<int,SYCLHostAllocator> delay_list; 
  /// current number of delays, increase one for each acceptance, reset to 0 after updating Ainv
  int delay_count;

  /** @ingroup Resources
   *  @{ */
  // SYCL queue
  //std::unique_ptr<sycl::queue> sycl_handles_;
  sycl::queue* m_queue = nullptr;
  // use it for async operations while interacting with the callers
  sycl::event m_event;
  /// crowd scope memory resource
  std::unique_ptr<MatrixDelayedUpdateSYCLMultiWalkerMem> mw_mem_;
  /**}@ */

  inline void waitStream()
  {
    //m_event.wait();
  }

  /** ensure no previous delay left.
   *  This looks like it should be an assert
   */
  inline void guard_no_delay() const
  {
    if (delay_count != 0)
      throw std::runtime_error("BUG: unexpected call sequence delay_count is not 0");
  }

  // check if the number of maximal delay is 1 (SM-1)
  // \todo rename this something containing delay.
  inline bool isSM1() const { return Binv_gpu.rows() == 1; }

  /** compute the row of up-to-date Ainv
   * @param Ainv inverse matrix
   * @param rowchanged the row id corresponding to the proposed electron
   */
  static void mw_prepareInvRow(const RefVectorWithLeader<This_t>& engines, const int rowchanged)
  {
    auto& engine_leader              = engines.getLeader();
    auto& prepare_inv_row_buffer_H2D = engine_leader.mw_mem_->prepare_inv_row_buffer_H2D;
    const int norb                   = engine_leader.get_psiMinv().rows();
    const int nw                     = engines.size();
    int& delay_count                 = engine_leader.delay_count;
    prepare_inv_row_buffer_H2D.resize(7 * nw);

    const int lda_Binv = engine_leader.Binv_gpu.cols();
    Matrix<Value*> ptr_buffer(prepare_inv_row_buffer_H2D.data(), 7, nw);
    for (int iw = 0; iw < nw; iw++)
    {
      This_t& engine    = engines[iw];
      auto& psiMinv     = engine.get_ref_psiMinv();
      ptr_buffer[0][iw] = psiMinv.device_data() + rowchanged * psiMinv.cols();
      ptr_buffer[1][iw] = engine.invRow.device_data();
      ptr_buffer[2][iw] = engine.U_gpu.data();
      ptr_buffer[3][iw] = engine.p_gpu.data();
      ptr_buffer[4][iw] = engine.Binv_gpu.data();
      ptr_buffer[5][iw] = engine.Binv_gpu.data() + delay_count * lda_Binv;
      ptr_buffer[6][iw] = engine.V_gpu.data();
    }

    prepare_inv_row_buffer_H2D.updateTo();

    Value** oldRow_mw_ptr  = updateRow_buffer_H2D.device_data();
    Value** invRow_mw_ptr  = updateRow_buffer_H2D.device_data() + nw;
    Value** U_mw_ptr       = updateRow_buffer_H2D.device_data() + nw * 2;
    Value** p_mw_ptr       = updateRow_buffer_H2D.device_data() + nw * 3;
    Value** Binv_mw_ptr    = updateRow_buffer_H2D.device_data() + nw * 4;
    Value** BinvRow_mw_ptr = updateRow_buffer_H2D.device_data() + nw * 5;
    Value** V_mw_ptr       = updateRow_buffer_H2D.device_data() + nw * 6;

    //copy rows
    const size_t nbytes = norb * sizeof(Value);
    std::vector<sycl::event> events(nw);
    for (int iw = 0; iw < nw; ++iw)
      events[iw] = m_queue->memcpy(invRow_nw_ptr[iw], oldRow_mw_ptr[iw], nbytes);
    m_queue->wait(events);

    const auto trans    = oneapi::mkl::transpose::trans;
    const auto nontrans = oneapi::mkl::transpose::nontrans;

    //Using group API with a group, note the use of &cone, &cminus_one, and &czero
    const Value cone      = Value(1.0);
    const Value cminusone = Value(-1.0);
    const Value czero     = Value(0.0);

    // multiply V (NxK) Binv(KxK) U(KxN) invRow right to the left
    //BLAS::gemv('T', norb, delay_count, cone, U_gpu.data(), norb, invRow.data(), 1, czero, p_gpu.data(), 1);
    auto e = syclBLAS::gemv_batched(*m_queue, trans, norb, delay_count, &cone, U_mw_ptr, norb, invRow_mw_ptr, 1, &czero,
                                    p_mw_ptr, 1, nw);
    //BLAS::gemv('N', delay_count, delay_count, -cone, Binv.data(), lda_Binv, p.data(), 1, czero, Binv[delay_count], 1);
    e = syclBLAS::gemv_batched(*m_queue, nontrans, delay_count, delay_count, &cminusone, Binv_mw_ptr, lda_Binv,
                               p_mw_ptr, 1, &czero, BinvRow_mw_ptr, 1, nw, {e});
    //BLAS::gemv('N', norb, delay_count, cone, V.data(), norb, Binv[delay_count], 1, cone, invRow.data(), 1);
    syclBLAS::gemv_batched(*m_queue, nontranse, norb, delay_count, &cone, V_mw_ptr, norb, BinvRow_mw_ptr, 1, &cone,
                           invRow_mw_ptr, 1, nw, {e}).wait();
    //no need to wait in a perfect world

    engine_leader.invRow_id = rowchanged;
  }

  /** Do complete row updates
   *  many of these const arguments provide pointers or references
   *  somwhere in here is an update that doesn't get where it belongs resulting in a 0
   *  gradient later.
   *  Sad example of OpenMP target code that is far from clear and a poor substitute for a
   *  clear CPU reference implementation.
   *
   *  \param[in] engines
   *  \param[in] rowchanged
   *  \param[in] psiM_g_list        device ptrs
   *  \param[in] psiM_l_list        device ptrs
   *  \param[in] isAccepted         bool but wait some lists are also filtered
   *  \param[in] phi_vgl_v_dev_ptr  device ptr
   *  \param[in] phi_vgl_stride     size of each "vector" in phi_vgl_v
   *  \param[inout] ratios
   */
  static void mw_updateRow(const RefVectorWithLeader<This_t>& engines,
                           const int rowchanged,
                           const std::vector<Value*>& psiM_g_list,
                           const std::vector<Value*>& psiM_l_list,
                           const std::vector<bool>& isAccepted,
                           const Value* phi_vgl_v_dev_ptr,
                           const size_t phi_vgl_stride,
                           const std::vector<Value>& ratios)
  {
    auto& engine_leader = engines.getLeader();
    engine_leader.guard_no_delay();

    const size_t n_accepted = psiM_g_list.size();
#ifndef NDEBUG
    size_t n_true = std::count_if(isAccepted.begin(), isAccepted.end(), [](bool accepted) { return accepted; });
    assert(n_accepted == n_true);
#endif
    if (n_accepted == 0)
      return;

    auto& updateRow_buffer_H2D = engine_leader.mw_mem_->updateRow_buffer_H2D;
    auto& mw_temp              = engine_leader.mw_mem_->mw_temp;
    auto& mw_rcopy             = engine_leader.mw_mem_->mw_rcopy;
    const int norb             = engine_leader.get_ref_psiMinv().rows();
    const int lda              = engine_leader.get_ref_psiMinv().cols();

    mw_temp.resize(norb * n_accepted);
    mw_rcopy.resize(norb * n_accepted);
    updateRow_buffer_H2D.resize((sizeof(Value*) * 8 + sizeof(Value)) * n_accepted);

    // to handle T** of Ainv, psi_v, temp, rcopy
    Matrix<Value*> ptr_buffer(updateRow_buffer_H2D.data(), 8, n_accepted);
    // ratios are special
    auto& c_ratio_inv = engine_leader.mw_mem_->mw_ratio;

    for (int iw = 0, count = 0; iw < isAccepted.size(); iw++)
      if (isAccepted[iw])
      {
        ptr_buffer[0][count] = engines[iw].get_ref_psiMinv().device_data();
        ptr_buffer[1][count] = const_cast<Value*>(phi_vgl_v_dev_ptr + norb * iw);
        ptr_buffer[2][count] = mw_temp.device_data() + norb * count;
        ptr_buffer[3][count] = mw_rcopy.device_data() + norb * count;
        ptr_buffer[4][count] = psiM_g_list[count];
        ptr_buffer[5][count] = psiM_l_list[count];
        ptr_buffer[6][count] = phi_vgl_v_dev_ptr + phi_vgl_stride + norb * 3 * iw;
        ptr_buffer[7][count] = phi_vgl_v_dev_ptr + phi_vgl_stride * 4 + norb * iw;

        c_ratio_inv[count] = Value(-1) / ratios[iw];
        count++;
      }

    updateRow_buffer_H2D.updateTo();

    {
      Value** Ainv_mw_ptr   = updateRow_buffer_H2D.device_data();
      Value** phiV_mw_ptr   = updateRow_buffer_H2D.device_data() + n_accepted;
      Value** temp_mw_ptr   = updateRow_buffer_H2D.device_data() + n_accepted * 2;
      Value** rcopy_mw_ptr  = updateRow_buffer_H2D.device_data() + n_accepted * 3;
      Value** dpsiM_mw_out  = updateRow_buffer_H2D.device_data() + n_accepted * 4;
      Value** d2psiM_mw_out = updateRow_buffer_H2D.device_data() + n_accepted * 5;
      Value** dpsiM_mw_in   = updateRow_buffer_H2D.device_data() + n_accepted * 6;
      Value** d2psiM_mw_in  = updateRow_buffer_H2D.device_data() + n_accepted * 7;

      const auto trans  = oneapi::mkl::transpose::trans;
      const Value cone  = T(1.0);
      const Value czero = T(0.0);

      auto e = syclBLAS::gemv_batched(*m_queue, trans, norb, norb, &cone, Ainv_mw_ptr, lda, phiV_mw_ptr, 1, &czero,
                                      temp_mw_ptr, 1, n_accepted);

      copyAinvRow_saveGL<64>(*m_queue, rowchanged, norb, Ainv_mw_ptr, lda, temp_mw_ptr, rcopy_mw_ptr, dpsiM_mw_in,
                             d2psiM_mw_in, dpsiM_mw_out, d2psiM_mw_out, n_accepted);

      syclBLAS::ger_batched(*m_queue, norb, norb, c_ratio_inv.data(), rcopy_mw_ptr, 1, temp_mw_ptr, 1, Ainv_mw_ptr, lda,
                            n_accepted).wait()
    }
  }

public:
  /// default constructor
  MatrixDelayedUpdateSYCL() : invRow_id(-1), delay_count(0) {}

  MatrixDelayedUpdateSYCL(const MatrixDelayedUpdateSYCL&) = delete;

  /** resize the internal storage
   * @param norb number of electrons/orbitals
   * @param delay, maximum delay 0<delay<=norb
   */
  inline void resize(int norb, int delay)
  {
    V_gpu.resize(delay, norb);
    U_gpu.resize(delay, norb);
    p_gpu.resize(delay);
    tempMat_gpu.resize(norb, delay);
    Binv_gpu.resize(delay, delay);
    delay_list_gpu.resize(delay);
    invRow.resize(norb);
    psiMinv_.resize(norb, getAlignedSize<Value>(norb));
  }

  void createResource(ResourceCollection& collection) const {}

  void acquireResource(ResourceCollection& collection) {}

  void releaseResource(ResourceCollection& collection) {}

  inline void checkResourcesForTest() {}

  Value* getRow_psiMinv_offload(int row_id) { return psiMinv_.device_data() + row_id * psiMinv_.cols(); }

  // prepare invRow and compute the old gradients.
  template<typename GT>
  static void mw_evalGrad(const RefVectorWithLeader<This_t>& engines,
                          const std::vector<const Value*>& dpsiM_row_list,
                          const int rowchanged,
                          std::vector<GT>& grad_now)
  {
    auto& engine_leader = engines.getLeader();
    if (!engine_leader.isSM1())
      mw_prepareInvRow(engines, rowchanged);

    auto& evalGrad_buffer_H2D = engine_leader.mw_mem_->evalGrad_buffer_H2D;
    auto& grads_value_v       = engine_leader.mw_mem_->grads_value_v;

    const int nw = engines.size();
    evalGrad_buffer_H2D.resize(sizeof(Value*) * 2 * nw);
    Matrix<const Value*> ptr_buffer(evalGrad_buffer_H2D.data(), 2, nw);
    for (int iw = 0; iw < nw; iw++)
    {
      if (engine_leader.isSM1())
      {
        auto& psiMinv     = engines[iw].get_ref_psiMinv();
        ptr_buffer[0][iw] = psiMinv.device_data() + rowchanged * psiMinv.cols();
      }
      else
        ptr_buffer[0][iw] = engines[iw].invRow.device_data();
      ptr_buffer[1][iw] = dpsiM_row_list[iw];
    }

    evalGrad_buffer_H2D.updateTo();

    if (grads_value_v.rows() != nw || grads_value_v.cols() != GT::Size)
      grads_value_v.resize(nw, GT::Size);

    const Value** invRow_ptr    = evalGrad_buffer_H2D.device_data();
    const Value** dpsiM_row_ptr = evalGrad_buffer_H2D.device_data() + nw;

    const int norb = engine_leader.get_ref_psiMinv().rows();
    calcGradients(*m_queue, norb, invRow_ptr, dpsiM_row_ptr, grads_value_v.device_data(), nw).wait();
    grads_value_v.updateFrom(); //
    //m_queue->memcpy(grads_value_v.data(), grads_value_v.device_data(), grads_value_v.size(),e).wait();

    for (int iw = 0; iw < nw; iw++)
      grad_now[iw] = {grads_value_v[iw][0], grads_value_v[iw][1], grads_value_v[iw][2]};
  }

  /** Update the "local" psiMinv_ on the device.
   *  Side Effect Transfers:
   *  * phiV is left on host side in the single methods so it must be transferred to device
   *  * psiMinv_ is transferred back to host since single calls from QMCHamitonian and others
   *  * expect it to be.
   *
   *  Forced to use OpenMP target since resources are banned for single walker functions APIs
   *  and the acquireRelease pattern for a single DDB was removed by #3324
   */
  template<typename VVT>
  void updateRow(int rowchanged, const VVT& phiV, FullPrecValue c_ratio_in)
  {
    guard_no_delay();
    auto& Ainv = psiMinv_;
    const int norb = Ainv.rows();
    const int lda  = Ainv.cols();

    if (phiV_temp.size() < norb)
    {//all device Vectors
      phiV_temp.resize(norb);
      temp.resize(norb);
      rcopy.resize(norb);
    }

    //Not sure what phiV's allocator is, can bypass this
    //
    Value* phiV_ptr  = phiV_temp.data();
    Value* Ainv_ptr  = Ainv.device_data();
    Value* temp_ptr  = temp.data();
    Value* rcopy_ptr = rcopy.data();

    //Assume phiV is on the host
    sycl::event e = m_queue->memcpy(phiV_ptr, phiV.data(), phiV.size()); 

    // update the inverse matrix
    constexpr Value cone(1), czero(0);

    syclBLAS::gemv(*m_queue, oneapi::mkl::transpose::trans, norb, norb, 
                   cone, Ainv_ptr, lda, phiV_ptr, 1, czero, temp_ptr, 1, {e}).wait();

    m_queue->parallel_for(sycl::range<1>{size_t(norb)}, 
        [=](sycl::id<1> tid) 
        {
          if(id==0)  temp_ptr[rowchanged] -= cone;
          rcopy_ptr[tid] = Ainv_ptr[rowchanged * lda + i];
        }).wait();

    syclBLAS::ger(*m_queue, norb, norb, static_cast<Value>(FullPrecValue(-1) / c_ratio_in), rcopy_ptr, 1, temp_ptr, 1,
                  Ainv_ptr, lda);
    
    Ainv.updateFrom();

  }

  /** Accept or Reject row updates
   *  many of these const arguments provide pointers or references
   *  to objects that do get modified.
   *  \param[in] engines
   *  \param[in] rowchanged
   *  \param[in] psiM_g_list
   *  \param[in] psiM_l_list
   *  \param[in] isAccepted
   *  \param[in] phi_vgl_v_dev_ptr
   *  \param[in] phi_vgl_stride     size of each "vector" in phi_vgl_v
   *  \param[inout] ratios
   */
  static void mw_accept_rejectRow(const RefVectorWithLeader<This_t>& engines,
                                  const int rowchanged,
                                  const std::vector<Value*>& psiM_g_list,
                                  const std::vector<Value*>& psiM_l_list,
                                  const std::vector<bool>& isAccepted,
                                  const Value* phi_vgl_v_dev_ptr,
                                  const size_t phi_vgl_stride,
                                  const std::vector<Value>& ratios)
  {
    auto& engine_leader = engines.getLeader();
    // invRow consumed, mark invRow_id unset
    engine_leader.invRow_id = -1;

    if (engine_leader.isSM1())
    {
      mw_updateRow(engines, rowchanged, psiM_g_list, psiM_l_list, isAccepted, phi_vgl_v_dev_ptr, phi_vgl_stride,
                   ratios);
      return;
    }

    auto& accept_rejectRow_buffer_H2D = engine_leader.mw_mem_->accept_rejectRow_buffer_H2D;
    int& delay_count                  = engine_leader.delay_count;
    const int lda_Binv                = engine_leader.Binv_gpu.cols();
    const int norb                    = engine_leader.get_psiMinv().rows();
    const int lda                     = engine_leader.get_psiMinv().cols();
    const int nw                      = engines.size();
    const int n_accepted              = psiM_g_list.size();

    accept_rejectRow_buffer_H2D.resize(14);

    Matrix<Value*> ptr_buffer(accept_rejectRow_buffer_H2D.data(), 14, nw);
    Value* c_ratio_inv = mw_ratio.data();
    for (int iw = 0, count_accepted = 0, count_rejected = 0; iw < nw; iw++)
    {
      This_t& engine = engines[iw];
      if (isAccepted[iw])
      {
        ptr_buffer[0][count_accepted]  = engine.psiMinv_.device_data() + lda * rowchanged;
        ptr_buffer[1][count_accepted]  = engine.V_gpu.data();
        ptr_buffer[2][count_accepted]  = engine.U_gpu.data() + norb * delay_count;
        ptr_buffer[3][count_accepted]  = engine.p_gpu.data();
        ptr_buffer[4][count_accepted]  = engine.Binv_gpu.data();
        ptr_buffer[5][count_accepted]  = engine.Binv_gpu.data() + delay_count * lda_Binv;
        ptr_buffer[6][count_accepted]  = engine.Binv_gpu.data() + delay_count;
        ptr_buffer[7][count_accepted]  = reinterpret_cast<Value*>(engine.delay_list_gpu.data());
        ptr_buffer[8][count_accepted]  = engine.V_gpu.data() + norb * delay_count;
        ptr_buffer[9][count_accepted]  = const_cast<Value*>(phi_vgl_v_dev_ptr + norb * iw);
        ptr_buffer[10][count_accepted] = const_cast<Value*>(phi_vgl_v_dev_ptr + phi_vgl_stride + norb * 3 * iw);
        ptr_buffer[11][count_accepted] = const_cast<Value*>(phi_vgl_v_dev_ptr + phi_vgl_stride * 4 + norb * iw);
        ptr_buffer[12][count_accepted] = psiM_g_list[count_accepted];
        ptr_buffer[13][count_accepted] = psiM_l_list[count_accepted];
        c_ratio_inv[count_accepted]    = Value(1) / ratios[iw];
        count_accepted++;
      }
      else
      {
        ptr_buffer[0][n_accepted + count_rejected] = engine.get_ref_psiMinv().device_data() + lda * rowchanged;
        ptr_buffer[1][n_accepted + count_rejected] = engine.V_gpu.data();
        ptr_buffer[2][n_accepted + count_rejected] = engine.U_gpu.data() + norb * delay_count;
        ptr_buffer[3][n_accepted + count_rejected] = engine.p_gpu.data();
        ptr_buffer[4][n_accepted + count_rejected] = engine.Binv_gpu.data();
        ptr_buffer[5][n_accepted + count_rejected] = engine.Binv_gpu.data() + delay_count * lda_Binv;
        ptr_buffer[6][n_accepted + count_rejected] = engine.Binv_gpu.data() + delay_count;
        ptr_buffer[7][n_accepted + count_rejected] = reinterpret_cast<Value*>(engine.delay_list_gpu.data());
        ptr_buffer[8][n_accepted + count_rejected] = engine.V_gpu.data() + norb * delay_count;
        count_rejected++;
      }
    }

    Value** invRow_mw_ptr   = accept_rejectRow_buffer_H2D.device_data();
    Value** V_mw_ptr        = accept_rejectRow_buffer_H2D.device_data() + nw;
    Value** U_row_mw_ptr    = accept_rejectRow_buffer_H2D.device_data() + nw * 2;
    Value** p_mw_ptr        = accept_rejectRow_buffer_H2D.device_data() + nw * 3;
    Value** Binv_mw_ptr     = accept_rejectRow_buffer_H2D.device_data() + nw * 4;
    Value** BinvRow_mw_ptr  = accept_rejectRow_buffer_H2D.device_data() + nw * 5;
    Value** BinvCol_mw_ptr  = accept_rejectRow_buffer_H2D.device_data() + nw * 6;
    int** delay_list_mw_ptr = reinterpret_cast<int**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 7);
    Value** V_row_mw_ptr    = accept_rejectRow_buffer_H2D.device_data() + nw * 8;
    Value** phiV_mw_ptr     = accept_rejectRow_buffer_H2D.device_data() + nw * 9;
    Value** dpsiM_mw_in     = accept_rejectRow_buffer_H2D.device_data() + nw * 10;
    Value** d2psiM_mw_in    = accept_rejectRow_buffer_H2D.device_data() + nw * 11;
    Value** dpsiM_mw_out    = accept_rejectRow_buffer_H2D.device_data() + nw * 12;
    Value** d2psiM_mw_out   = accept_rejectRow_buffer_H2D.device_data() + nw * 13;
    Value* ratio_inv_mw_ptr = mw_ratio.data();

    const size_t nbytes = norb * sizeof(Value);
    std::vector<sycl::event> events(nw);

    //std::copy_n(Ainv[rowchanged], norb, V[delay_count]);
    for (int iw = 0; iw < nw; ++iw)
      events[iw] = m_queue->memcpy(V_row_mw_ptr[iw], invRow_nw_ptr[iw], nbytes);
    m_queue->wait(events);

    const auto trans  = oneapi::mkl::transpose::trans;
    const Value cminusone  = T(-1.0);
    const Value cone  = T(1.0);
    const Value czero = T(0.0);


    // handle accepted walkers
    // the new Binv is [[X Y] [Z sigma]]
    //BLAS::gemv('T', norb, delay_count + 1, cminusone, V.data(), norb, psiV.data(), 1, czero, p.data(), 1);
    sycl::event e =syclBLAS::gemv_batched(*m_queue, trans, norb, delay_count, &cminusone, V_mw_ptr,
                           norb, phiV_mw_ptr, 1, &czero, p_mw_ptr, 1, n_accepted);
    // Y
    //BLAS::gemv('T', delay_count, delay_count, sigma, Binv.data(), lda_Binv, p.data(), 1, czero, Binv.data() + delay_count,
    //           lda_Binv);
    auto success = syclBLAS::gemv_batched(*m_queue, trans, delay_count, delay_count, 
                                          ratio_inv_mw_ptr, n_accepted, Binv_mw_ptr,
                                          lda_Binv, p_mw_ptr, 1, &czero, BinvCol_mw_ptr, lda_Binv,
                                          n_accepted,{e});
    // X
    //BLAS::ger(delay_count, delay_count, cone, Binv[delay_count], 1, Binv.data() + delay_count, lda_Binv,
    //          Binv.data(), lda_Binv);
    syclBLAS::ger_batched(*m_queue, delay_count, delay_count, &cone, BinvRow_mw_ptr, 1,
                          BinvCol_mw_ptr, lda_Binv, Binv_mw_ptr, lda_Binv, n_accepted).wait();

    add_delay_list_save_sigma_VGL(*m_queue, delay_list_mw_ptr, rowchanged, delay_count,
                                  Binv_mw_ptr, lda_Binv, ratio_inv_mw_ptr, phiV_mw_ptr,
                                  dpsiM_mw_in, d2psiM_mw_in, U_row_mw_ptr, dpsiM_mw_out,
                                  d2psiM_mw_out, norb, n_accepted, nw).wait();
    delay_count++;

    // update Ainv when maximal delay is reached
    if (delay_count == lda_Binv)
      mw_updateInvMat(engines);
  }

  /** update the full Ainv and reset delay_count
   * @param Ainv inverse matrix
   */
  static void mw_updateInvMat(const RefVectorWithLeader<This_t>& engines)
  {
    auto& engine_leader = engines.getLeader();
    int& delay_count    = engine_leader.delay_count;
    if (delay_count == 0)
      return;
    auto& updateInv_buffer_H2D = engine_leader.mw_mem_->updateInv_buffer_H2D;
    const int norb             = engine_leader.get_psiMinv().rows();
    const int lda              = engine_leader.get_psiMinv().cols();
    const int nw               = engines.size();
    updateInv_buffer_H2D.resize(6 * nw);

    Matrix<Value*> ptr_buffer(updateInv_buffer_H2D.data(), 6, nw);
    for (int iw = 0; iw < nw; iw++)
    {
      This_t& engine    = engines[iw];
      ptr_buffer[0][iw] = engine.U_gpu.data();
      ptr_buffer[1][iw] = engine.get_ref_psiMinv().device_data();
      ptr_buffer[2][iw] = engine.tempMat_gpu.data();
      ptr_buffer[3][iw] = reinterpret_cast<Value*>(engine.delay_list_gpu.data());
      ptr_buffer[4][iw] = engine.V_gpu.data();
      ptr_buffer[5][iw] = engine.Binv_gpu.data();
    }

    //cudaErrorCheck(cudaMemcpyAsync(updateInv_buffer_H2D.device_data(), updateInv_buffer_H2D.data(),
    //                               updateInv_buffer_H2D.size(), cudaMemcpyHostToDevice, hstream),
    //               "cudaMemcpyAsync updateInv_buffer_H2D failed!");

    Value** U_mw_ptr        = updateInv_buffer_H2D.device_data();
    Value** Ainv_mw_ptr     = updateInv_buffer_H2D.device_data() + nw;
    Value** tempMat_mw_ptr  = updateInv_buffer_H2D.device_data() + nw * 2;
    //DON"T LIKE THIS
    int** delay_list_mw_ptr = reinterpret_cast<int**>(updateInv_buffer_H2D.device_data() + nw * 3);
    Value** V_mw_ptr        = updateInv_buffer_H2D.device_data() + nw * 4;
    Value** Binv_mw_ptr     = updateInv_buffer_H2D.device_data() + nw * 5;

    /*
    if (delay_count == 1)
    {
      // this is a special case invoking the Fahy's variant of Sherman-Morrison update.
      // Only use the first norb elements of tempMat as a temporal array
      BLAS::gemv('T', norb, norb, cone, Ainv.data(), norb, U[0], 1, czero, temp.data(), 1);
      temp[delay_list[0]] -= cone;
      BLAS::ger(norb, norb, -Binv[0][0], V[0], 1, temp.data(), 1, Ainv.data(), norb);
    }
    else
*/
    {
      const auto trans    = oneapi::mkl::transpose::trans;
      const auto nontrans = oneapi::mkl::transpose::nontrans;

      //Using group API with a group, note the use of &cone, &cminus_one, and &czero
      const Value cone       = T(1.0);
      const Value cminus_one = T(-1.0);
      const Value czero      = T(0.0);

      const int lda_Binv = engine_leader.Binv_gpu.cols();
      constexpr Value cone(1), czero(0), cminusone(-1);
      mklBLAS::gemm_batched(*m_queue,&trans, &nontrans, delay_count, norb, norb, &cone,
                                     U_mw_ptr, norb, Ainv_mw_ptr, lda, &czero, tempMat_mw_ptr, lda_Binv, nw).wait();

      //cudaErrorCheck(SYCL::applyW_batched(hstream, delay_list_mw_ptr, delay_count, tempMat_mw_ptr, lda_Binv, nw),
      auto e =applyW_batched(*m_queue,delay_list_mw_ptr, delay_count, tempMat_mw_ptr, lda_Binv, nw);

      e = mklBLAS::gemm_batched(*m_queue,&nontrans, &nontrans, 
                            norb, delay_count, delay_count, &cone,
                            V_mw_ptr, norb, Binv_mw_ptr, lda_Binv, &czero, U_mw_ptr, norb, nw,{e});

      mklBLAS::gemm_batched(*m_queue,&nontrans, &nontrans, 
          norb, norb, delay_count, &cminusone,
          U_mw_ptr, norb, tempMat_mw_ptr, lda_Binv, &cone, Ainv_mw_ptr, lda, nw,{e}).wait();
    }
    delay_count = 0;
  }

  inline void print_Ainv(const RefVector<This_t>& engines)
  {
    for (This_t& engine : engines)
    {
      std::cout << "debug Ainv host  " << engine.get_psiMinv()[0][0] << " " << engine.get_psiMinv()[0][1] << " "
                << engine.psiMinv[1][0] << " " << engine.psiMinv[1][1] << std::endl;
      auto* temp_ptr = engine.psiMinv.data();
      PRAGMA_OFFLOAD("omp target update from(temp_ptr[:psiMinv_.size()])")
      std::cout << "debug Ainv devi  " << engine.psiMinv[0][0] << " " << engine.psiMinv[0][1] << " "
                << engine.psiMinv[1][0] << " " << engine.psiMinv[1][1] << std::endl;
    }
  }

  /** return invRow host or device pointers based on on_host request
   * prepare invRow if not already.
   */
  static std::vector<const Value*> mw_getInvRow(const RefVectorWithLeader<This_t>& engines,
                                                const int row_id,
                                                bool on_host)
  {
    auto& engine_leader = engines.getLeader();
    if (engine_leader.isSM1())
      engine_leader.waitStream();
    else if (engine_leader.invRow_id != row_id)
    {
      // this can be skipped if mw_evalGrad gets called already.
      mw_prepareInvRow(engines, row_id);
      engine_leader.waitStream();
    }

    const size_t ncols = engines.getLeader().get_psiMinv().cols();
    const size_t nw    = engines.size();
    std::vector<const Value*> row_ptr_list;
    row_ptr_list.reserve(nw);
    if (on_host)
    {
      // copy values to host and return host pointer
      for (This_t& engine : engines)
        if (engine_leader.isSM1())
        {
          auto* ptr = engine.get_ref_psiMinv().data();
          PRAGMA_OFFLOAD("omp target update from(ptr[row_id * ncols : ncols])")
          row_ptr_list.push_back(ptr + row_id * ncols);
        }
        else
        {
          auto* ptr = engine.invRow.data();
          PRAGMA_OFFLOAD("omp target update from(ptr[:engine.invRow.size()])")
          row_ptr_list.push_back(ptr);
        }
    }
    else
    {
      // return device pointer
      for (This_t& engine : engines)
        if (engine_leader.isSM1())
          row_ptr_list.push_back(engine.get_ref_psiMinv().device_data() + row_id * ncols);
        else
          row_ptr_list.push_back(engine.invRow.device_data());
    }
    return row_ptr_list;
  }

  static void mw_transferAinv_D2H(const RefVectorWithLeader<This_t>& engines)
  {
    auto& engine_leader = engines.getLeader();
    auto& hstream       = engine_leader.cuda_handles_->hstream;
    engine_leader.guard_no_delay();

    for (This_t& engine : engines)
      cudaErrorCheck(cudaMemcpyAsync(engine.get_ref_psiMinv().data(), engine.get_ref_psiMinv().device_data(),
                                     engine.get_psiMinv().size() * sizeof(Value), cudaMemcpyDeviceToHost, hstream),
                     "cudaMemcpyAsync Ainv failed!");
    engine_leader.waitStream();
  }

  auto& getLAhandles()
  {
    if (!cuda_handles_)
      throw std::logic_error("attempted to get null cuda_handles_, this is developer logic error");
    return *cuda_handles_;
  }
};
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_MATRIX_DELAYED_UPDATE_SYCL_H
