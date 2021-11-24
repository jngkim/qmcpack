///////////////////////////////////////////////////////////////////////////////////////
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
#include "QMCWaveFunctions/detail/SYCL/sycl_determinant_helper.hpp"
#include "DualAllocatorAliases.hpp"
#include "DiracMatrixComputeSYCL.hpp"
#include "ResourceCollection.h"
#include "WaveFunctionTypes.hpp"

namespace qmcplusplus
{

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

  //use the same allocator as DetInverter
  template<typename DT>
  using PinnedDualAllocator = typename DetInverter::template OffloadPinnedAllocator<DT>;
  template<typename DT>
  using DualVector = Vector<DT, PinnedDualAllocator<DT>>;
  template<typename DT>
  using DualMatrix = Matrix<DT, PinnedDualAllocator<DT>>;

  //DeviceVector 
  template<typename DT>
  using DeviceVector = Vector<DT, SYCLAllocator<DT>>;
  //DeviceMatrix 
  template<typename DT>
  using DeviceMatrix = Matrix<DT, SYCLAllocator<DT>>;
  template<typename DT>
  using PinnedHostVector = Vector<DT, SYCLHostAllocator<DT>>;
  template<typename DT>
  using PinnedHostMatrix = Matrix<DT, SYCLHostAllocator<DT>>;

  struct MatrixDelayedUpdateSYCLMultiWalkerMem : public Resource
  {
    // multi walker of grads for transfer needs. 
    DualMatrix<Value> grads_value_v;
    //Two containers for each buffer required due to MKL APIs
    // mw_updateRow pointer buffer
    PinnedHostMatrix<Value*> updateRow_buffer_H2D;
    PinnedHostMatrix<const Value*> updateRow_buffer_H2D_C;
    // mw_prepareInvRow pointer buffer
    PinnedHostMatrix<Value*> prepare_inv_row_buffer_H2D;
    PinnedHostMatrix<const Value*> prepare_inv_row_buffer_H2D_C;
    // mw_accept_rejectRow pointer buffer
    PinnedHostMatrix<Value*> accept_rejectRow_buffer_H2D;
    PinnedHostMatrix<const Value*> accept_rejectRow_buffer_H2D_C;
    // mw_updateInv pointer buffer
    PinnedHostMatrix<Value*> updateInv_buffer_H2D;
    PinnedHostMatrix<const Value*> updateInv_buffer_H2D_C;
    // mw_evalGrad pointer buffer
    PinnedHostVector<Value*> evalGrad_buffer_H2D; 
    /// scratch space for rank-1 update
    DeviceVector<Value> mw_temp;
    // scratch space for keeping one row of Ainv
    DeviceVector<Value> mw_rcopy;
    // ratios 
    PinnedHostVector<Value> mw_ratio;
    // delay list: Need to analyze the performance impact as it is modified on a device
    PinnedHostVector<int*> mw_delay_list;

    MatrixDelayedUpdateSYCLMultiWalkerMem() : Resource("MatrixDelayedUpdateSYCLMultiWalkerMem") {}

    MatrixDelayedUpdateSYCLMultiWalkerMem(const MatrixDelayedUpdateSYCLMultiWalkerMem&)
        : MatrixDelayedUpdateSYCLMultiWalkerMem()
    {}

    Resource* makeClone() const override { return new MatrixDelayedUpdateSYCLMultiWalkerMem(*this); }
  };

  const DualMatrix<Value>& get_psiMinv() const { return psiMinv_; }
  DualMatrix<Value>& get_ref_psiMinv() { return psiMinv_; }

private:
  /// inverter
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
  Vector<int,SYCLHostAllocator<int>> delay_list;
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

  inline sycl::queue* get_queue() 
  {
    if(m_queue==nullptr) m_queue=get_default_queue();
    return m_queue;
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
    auto& ptr_buffer                 = engine_leader.mw_mem_->prepare_inv_row_buffer_H2D; 
    auto& cptr_buffer                = engine_leader.mw_mem_->prepare_inv_row_buffer_H2D_C;
    const int norb                   = engine_leader.get_psiMinv().rows();
    const int nw                     = engines.size();
    int& delay_count                 = engine_leader.delay_count;
    const int lda_Binv               = engine_leader.Binv_gpu.cols();

    sycl::queue& lead_q(*engine_leader.get_queue()); 
    std::vector<sycl::event> events(nw);

    enum {C_invRow=0, C_p=1, C_Binv=2, C_BinvRow=3, C_U=4, C_V=5, C_old=6};  

    ptr_buffer.resize(4,nw);
    cptr_buffer.resize(7,nw);

    for (int iw = 0; iw < nw; iw++)
    {
      This_t& engine    = engines[iw];
      auto& psiMinv     = engine.get_ref_psiMinv();

      //const Value*
      cptr_buffer[C_invRow ][iw] = engine.invRow.device_data();
      cptr_buffer[C_p      ][iw] = engine.p_gpu.data();
      cptr_buffer[C_Binv   ][iw] = engine.Binv_gpu.data();
      cptr_buffer[C_BinvRow][iw] = engine.Binv_gpu.data() + delay_count * lda_Binv;
      cptr_buffer[C_U      ][iw] = engine.U_gpu.data();       
      cptr_buffer[C_V      ][iw] = engine.V_gpu.data();
      cptr_buffer[C_old    ][iw] = psiMinv.device_data() + rowchanged * psiMinv.cols();

      ptr_buffer [C_invRow ][iw] = engine.invRow.device_data();
      ptr_buffer [C_p      ][iw] = engine.p_gpu.data();
      ptr_buffer [C_Binv   ][iw] = engine.Binv_gpu.data();
      ptr_buffer [C_BinvRow][iw] = engine.Binv_gpu.data() + delay_count * lda_Binv;
    }

    { 
      const size_t nbytes = norb * sizeof(Value);
      for (int iw = 0; iw < nw; ++iw)
      {
        events[iw] = lead_q.memcpy(ptr_buffer[C_invRow][iw], cptr_buffer[C_old][iw], nbytes);
      }
    }

    constexpr auto trans    = oneapi::mkl::transpose::trans;
    constexpr auto nontrans = oneapi::mkl::transpose::nontrans;

    //Using group API with a group, note the use of &cone, &cminus_one, and &czero
    constexpr Value cone(1);
    constexpr Value cminusone(-1);
    constexpr Value czero{};

    // multiply V (NxK) Binv(KxK) U(KxN) invRow right to the left
    //BLAS::gemv('T', norb, delay_count, cone, U_gpu.data(), norb, invRow.data(), 1, czero, p_gpu.data(), 1);
    auto e = syclBLAS::gemv_batched(lead_q, trans, norb, delay_count, &cone, cptr_buffer[C_U], norb, 
                                    cptr_buffer[C_invRow], 1, &czero, ptr_buffer[C_p], 1, nw, events);
    //BLAS::gemv('N', delay_count, delay_count, -cone, Binv.data(), lda_Binv, p.data(), 1, czero, Binv[delay_count], 1);
    e = syclBLAS::gemv_batched(lead_q, nontrans, delay_count, delay_count, &cminusone, cptr_buffer[C_Binv], lda_Binv, 
                               cptr_buffer[C_p], 1, &czero, ptr_buffer[C_BinvRow], 1, nw, {e});
    //BLAS::gemv('N', norb, delay_count, cone, V.data(), norb, Binv[delay_count], 1, cone, invRow.data(), 1);
    syclBLAS::gemv_batched(lead_q, nontrans, norb, delay_count, &cone, cptr_buffer[C_V], norb, 
                           cptr_buffer[C_BinvRow], 1, &cone, ptr_buffer[C_invRow], 1, nw, {e}).wait();
    //complete gemv's 
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

    auto& ptr_buffer = engine_leader.mw_mem_->updateRow_buffer_H2D;
    auto& cptr_buffer = engine_leader.mw_mem_->updateRow_buffer_H2D_C;
    auto& mw_temp    = engine_leader.mw_mem_->mw_temp;
    auto& mw_rcopy   = engine_leader.mw_mem_->mw_rcopy;
    const int norb   = engine_leader.get_ref_psiMinv().rows();
    const int lda    = engine_leader.get_ref_psiMinv().cols();

    mw_temp.resize(norb * n_accepted);
    mw_rcopy.resize(norb * n_accepted);

    constexpr unsigned C_Ainv   = 0;
    constexpr unsigned C_dpsiM  = 1;
    constexpr unsigned C_d2psiM = 2;
    constexpr unsigned C_temp   = 3; //used only for Value*
    constexpr unsigned C_rcopy  = 4;
    constexpr unsigned C_phiV   = 3; //used only for const Value*

    ptr_buffer.resize(5,n_accepted);
    cptr_buffer.resize(4,n_accepted);

    auto& c_ratio_inv = engine_leader.mw_mem_->mw_ratio;

    for (int iw = 0, count = 0; iw < isAccepted.size(); iw++)
      if (isAccepted[iw])
      {
        ptr_buffer[C_Ainv][count]    = engines[iw].get_ref_psiMinv().device_data();
        ptr_buffer[C_dpsiM][count]   = psiM_g_list[count];
        ptr_buffer[C_d2psiM][count]  = psiM_l_list[count];
        ptr_buffer[C_temp][count]    = mw_temp.data() + norb * count;
        ptr_buffer[C_rcopy][count]   = mw_rcopy.data() + norb * count;

        cptr_buffer[C_Ainv][count]   = engines[iw].get_ref_psiMinv().device_data();
        cptr_buffer[C_phiV][count]   = phi_vgl_v_dev_ptr + norb * iw;
        cptr_buffer[C_dpsiM][count]  = phi_vgl_v_dev_ptr + phi_vgl_stride + norb * 3 * iw;
        cptr_buffer[C_d2psiM][count] = phi_vgl_v_dev_ptr + phi_vgl_stride * 4 + norb * iw;

        c_ratio_inv[count] = Value(-1) / ratios[iw];
        count++;
      }

    constexpr auto trans  = oneapi::mkl::transpose::trans;
    constexpr Value cone  = Value(1.0);
    constexpr Value czero = Value(0.0);

    sycl::queue& lead_q(*(engine_leader.m_queue));
    auto e = syclBLAS::gemv_batched(lead_q, trans, norb, norb, &cone, cptr_buffer[C_Ainv], lda, 
                                    cptr_buffer[C_phiV], 1, &czero, ptr_buffer[C_temp], 1, n_accepted);

    copyAinvRow_saveGL(lead_q, rowchanged, norb, cptr_buffer[C_Ainv], lda, 
                       ptr_buffer[C_temp], ptr_buffer[C_rcopy], 
                       cptr_buffer[C_dpsiM], cptr_buffer[C_d2psiM], 
                       ptr_buffer[C_dpsiM], ptr_buffer[C_d2psiM],  n_accepted);

    syclBLAS::ger_batched(lead_q, norb, norb, c_ratio_inv.data(), ptr_buffer[C_rcopy], 1, 
                          ptr_buffer[C_temp], 1, ptr_buffer[C_Ainv], lda, n_accepted).wait();
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
    delay_list.resize(delay);
    invRow.resize(norb);
    psiMinv_.resize(norb, getAlignedSize<Value>(norb));
  }

  void createResource(ResourceCollection& collection) const 
  {
    collection.addResource(std::make_unique<MatrixDelayedUpdateSYCLMultiWalkerMem>());
  }

  void acquireResource(ResourceCollection& collection) {
    auto res_ptr = dynamic_cast<MatrixDelayedUpdateSYCLMultiWalkerMem*>(collection.lendResource().release());
    if (!res_ptr)
      throw std::runtime_error(
          "MatrixUpdateOMPTarget::acquireResource dynamic_cast MatrixUpdateOMPTargetMultiWalkerMem failed");
    mw_mem_.reset(res_ptr);
  }

  void releaseResource(ResourceCollection& collection) { collection.takebackResource(std::move(mw_mem_)); }

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
    evalGrad_buffer_H2D.resize(2 * nw);

    if (engine_leader.isSM1())
    {
      for (int iw = 0; iw < nw; iw++)
      {
        auto& psiMinv     = engines[iw].get_ref_psiMinv();
        evalGrad_buffer_H2D[iw] = psiMinv.device_data() + rowchanged * psiMinv.cols();
      }
    }
    else
      for (int iw = 0; iw < nw; iw++)
        evalGrad_buffer_H2D[iw] = engines[iw].invRow.device_data();
    
    //FIXME: need to copy host pointer??? 
    std::copy_n(const_cast<Value**>(dpsiM_row_list.data()),nw,evalGrad_buffer_H2D.data()); //THIS IS NOT GOING TO WORK
    //evalGrad_buffer_H2D.updateTo();

    if (grads_value_v.rows() != nw || grads_value_v.cols() != GT::Size)
      grads_value_v.resize(nw, GT::Size);

    sycl::queue& lead_q(*(engine_leader.m_queue));
    const int norb = engine_leader.get_ref_psiMinv().rows();

    //simplify this
    /*const*/ Value** invRow_ptr    = evalGrad_buffer_H2D.data();
    /*const*/ Value** dpsiM_row_ptr = evalGrad_buffer_H2D.data() + nw;
    calcGradients(lead_q, norb, invRow_ptr, dpsiM_row_ptr, grads_value_v.device_data(), nw).wait();
    grads_value_v.updateFrom(); 
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
   *
   *  TESTED
   */
  template<typename VVT>
  void updateRow(int rowchanged, const VVT& phiV, FullPrecValue c_ratio_in)
  {
    guard_no_delay();
    auto& Ainv = psiMinv_;
    const int norb = Ainv.rows();
    const int lda  = Ainv.cols();

    if(m_queue==nullptr) m_queue=get_default_queue();

    if (temp.size() < lda)
    {
      temp.resize(lda);
      rcopy.resize(lda);
    }

    Value* Ainv_ptr  = Ainv.device_data();
    Value* temp_ptr  = temp.data();
    Value* rcopy_ptr = rcopy.data();

    sycl::event e=m_queue->memcpy(rcopy_ptr,phiV.data(),sizeof(Value)*norb);

    // update the inverse matrix
    constexpr Value cone(1), czero(0);

    syclBLAS::gemv(*m_queue, oneapi::mkl::transpose::trans, norb, norb, 
                   cone, Ainv_ptr, lda, rcopy_ptr, 1, czero, temp_ptr, 1,{e}).wait();

    m_queue->parallel_for(sycl::range<1>{size_t(norb)}, 
        [=](sycl::id<1> tid) 
        {
          if(tid==0)  temp_ptr[rowchanged] -= cone;
          rcopy_ptr[tid] = Ainv_ptr[rowchanged * lda + tid];
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

    auto& ptr_buffer    = engine_leader.mw_mem_->accept_rejectRow_buffer_H2D;
    auto& cptr_buffer   = engine_leader.mw_mem_->accept_rejectRow_buffer_H2D_C;
    auto& mw_delay_list = engine_leader.mw_mem_->mw_delay_list;

    int& delay_count                  = engine_leader.delay_count;
    const int lda_Binv                = engine_leader.Binv_gpu.cols();
    const int norb                    = engine_leader.get_psiMinv().rows();
    const int lda                     = engine_leader.get_psiMinv().cols();
    const int nw                      = engines.size();
    const int n_accepted              = psiM_g_list.size();

    ptr_buffer.resize(12,nw);
    cptr_buffer.resize(12,nw);
    mw_delay_list.resize(nw);

    constexpr unsigned C_invRow  = 0;
    constexpr unsigned C_V       = 1;
    constexpr unsigned C_U_row   = 2;
    constexpr unsigned C_p       = 3;
    constexpr unsigned C_Binv    = 4;
    constexpr unsigned C_BinvRow = 5;
    constexpr unsigned C_BinvCol = 6;
    constexpr unsigned C_V_row   = 7;
    constexpr unsigned C_phiV    = 8;
    constexpr unsigned C_dpsiM   = 9;
    constexpr unsigned C_d2psiM  = 10;

    Value* c_ratio_inv = engine_leader.mw_mem_->mw_ratio.data();
    for (unsigned iw = 0, count_accepted = 0, count_rejected = 0; iw < nw; iw++)
    {
      This_t& engine = engines[iw];
      if (isAccepted[iw])
      {
        ptr_buffer[C_invRow ][count_accepted] = engine.psiMinv_.device_data() + lda * rowchanged;
        ptr_buffer[C_V      ][count_accepted] = engine.V_gpu.data();
        ptr_buffer[C_U_row  ][count_accepted] = engine.U_gpu.data() + norb * delay_count;
        ptr_buffer[C_p      ][count_accepted] = engine.p_gpu.data();
        ptr_buffer[C_Binv   ][count_accepted] = engine.Binv_gpu.data();
        ptr_buffer[C_BinvRow][count_accepted] = engine.Binv_gpu.data() + delay_count * lda_Binv;
        ptr_buffer[C_BinvCol][count_accepted] = engine.Binv_gpu.data() + delay_count;
        ptr_buffer[C_V_row  ][count_accepted] = engine.V_gpu.data() + norb * delay_count;
        ptr_buffer[C_dpsiM  ][count_accepted] = psiM_g_list[count_accepted];
        ptr_buffer[C_d2psiM ][count_accepted] = psiM_l_list[count_accepted];
        mw_delay_list        [count_accepted] = engine.delay_list.data();

        cptr_buffer[C_V     ][count_accepted] = engine.V_gpu.data();
        cptr_buffer[C_p     ][count_accepted] = engine.p_gpu.data();
        cptr_buffer[C_Binv  ][count_accepted] = engine.Binv_gpu.data();
        cptr_buffer[C_phiV  ][count_accepted] = phi_vgl_v_dev_ptr + norb * iw;
        cptr_buffer[C_dpsiM ][count_accepted] = phi_vgl_v_dev_ptr + phi_vgl_stride + norb * 3 * iw;
        cptr_buffer[C_d2psiM][count_accepted] = phi_vgl_v_dev_ptr + phi_vgl_stride * 4 + norb * iw;

        c_ratio_inv[count_accepted]    = Value(1) / ratios[iw];
        count_accepted++;
      }
      else
      {
        const unsigned rejected=n_accepted + count_rejected;
        ptr_buffer[C_invRow ][rejected] = engine.get_ref_psiMinv().device_data() + lda * rowchanged;
        ptr_buffer[C_V      ][rejected] = engine.V_gpu.data();
        ptr_buffer[C_BinvRow][rejected] = engine.Binv_gpu.data() + delay_count * lda_Binv;
        ptr_buffer[C_BinvCol][rejected] = engine.Binv_gpu.data() + delay_count;
        ptr_buffer[C_V_row  ][rejected] = engine.V_gpu.data() + norb * delay_count;
        ptr_buffer[C_U_row  ][rejected] = engine.U_gpu.data() + norb * delay_count;
        ptr_buffer[C_p      ][rejected] = engine.p_gpu.data();
        ptr_buffer[C_Binv   ][rejected] = engine.Binv_gpu.data();
        mw_delay_list        [rejected] = engine.delay_list.data();

        cptr_buffer[C_V     ][rejected] = engine.V_gpu.data();
        cptr_buffer[C_p     ][rejected] = engine.p_gpu.data();
        cptr_buffer[C_Binv  ][rejected] = engine.Binv_gpu.data();

        count_rejected++;
      }
    }


    sycl::queue& lead_q(*(engine_leader.m_queue));
    std::vector<sycl::event> events(nw);
    {
      const size_t nbytes = norb * sizeof(Value);
      for (int iw = 0; iw < nw; ++iw)
        events[iw] = lead_q.memcpy(ptr_buffer[C_V_row][iw], ptr_buffer[C_invRow][iw], nbytes);
    }

    constexpr auto trans  = oneapi::mkl::transpose::trans;
    constexpr Value cminusone  = Value(-1.0);
    constexpr Value cone  = Value(1.0);
    constexpr Value czero = Value(0.0);

    const Value* ratio_inv_mw_ptr = engine_leader.mw_mem_->mw_ratio.data();

    // handle accepted walkers
    // the new Binv is [[X Y] [Z sigma]]
    //BLAS::gemv('T', norb, delay_count + 1, cminusone, V.data(), norb, psiV.data(), 1, czero, p.data(), 1);
    sycl::event e =syclBLAS::gemv_batched(lead_q, trans, norb, delay_count, &cminusone, 
                                          cptr_buffer[C_V], norb, cptr_buffer[C_phiV], 1, &czero, 
                                          ptr_buffer[C_p], 1, n_accepted, events);
    // Y
    //BLAS::gemv('T', delay_count, delay_count, sigma, Binv.data(), lda_Binv, p.data(), 1, czero, Binv.data() + delay_count,
    //           lda_Binv);
    auto success = syclBLAS::gemv_batched_alpha(lead_q, trans, delay_count, delay_count, 
                                                ratio_inv_mw_ptr, n_accepted, cptr_buffer[C_Binv],
                                                lda_Binv, cptr_buffer[C_p], 1, czero, ptr_buffer[C_BinvCol], lda_Binv,
                                                n_accepted,{e});
    // X
    //BLAS::ger(delay_count, delay_count, cone, Binv[delay_count], 1, Binv.data() + delay_count, lda_Binv,
    //          Binv.data(), lda_Binv);
    syclBLAS::ger_batched(lead_q, delay_count, delay_count, &cone, ptr_buffer[C_BinvRow], 1,
                          ptr_buffer[C_BinvCol], lda_Binv, ptr_buffer[C_Binv], lda_Binv, n_accepted).wait();

    add_delay_list_save_sigma_VGL(lead_q, mw_delay_list.data(), rowchanged, delay_count,
                                  ptr_buffer[C_Binv], lda_Binv, ratio_inv_mw_ptr, cptr_buffer[C_phiV],
                                  cptr_buffer[C_dpsiM], cptr_buffer[C_d2psiM], ptr_buffer[C_U_row], ptr_buffer[C_dpsiM],
                                  ptr_buffer[C_d2psiM], norb, n_accepted, nw).wait();
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
    auto& ptr_buffer    = engine_leader.mw_mem_->updateInv_buffer_H2D;
    auto& cptr_buffer   = engine_leader.mw_mem_->updateInv_buffer_H2D_C;
    auto& mw_delay_list = engine_leader.mw_mem_->mw_delay_list;

    const int norb   = engine_leader.get_psiMinv().rows();
    const int lda    = engine_leader.get_psiMinv().cols();
    const int nw     = engines.size();

    ptr_buffer.resize(5, nw);
    cptr_buffer.resize(5, nw);

    constexpr unsigned C_U       =0;
    constexpr unsigned C_Ainv    =1;
    constexpr unsigned C_tempMat =2;
    constexpr unsigned C_V       =3;
    constexpr unsigned C_Binv    =4;


    for (int iw = 0; iw < nw; iw++)
    {
      This_t& engine    = engines[iw];

      ptr_buffer[C_U      ][iw] = engine.U_gpu.data();
      ptr_buffer[C_Ainv   ][iw] = engine.get_ref_psiMinv().device_data();
      ptr_buffer[C_tempMat][iw] = engine.tempMat_gpu.data();
      ptr_buffer[C_V      ][iw] = engine.V_gpu.data();
      ptr_buffer[C_Binv   ][iw] = engine.Binv_gpu.data();

      cptr_buffer[C_U      ][iw] = engine.U_gpu.data();
      cptr_buffer[C_Ainv   ][iw] = engine.get_ref_psiMinv().device_data();
      cptr_buffer[C_tempMat][iw] = engine.tempMat_gpu.data();
      cptr_buffer[C_V      ][iw] = engine.V_gpu.data();
      cptr_buffer[C_Binv   ][iw] = engine.Binv_gpu.data();

      mw_delay_list[iw] = engine.delay_list.data();
    }

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
      sycl::queue& lead_q(*(engine_leader.m_queue));
      constexpr auto trans    = oneapi::mkl::transpose::trans;
      constexpr auto nontrans = oneapi::mkl::transpose::nontrans;
      constexpr Value cone(1), czero(0), cminusone(-1);

      const int lda_Binv = engine_leader.Binv_gpu.cols();

      syclBLAS::gemm_batched(lead_q, trans, nontrans, delay_count, norb, norb, &cone,
                             cptr_buffer[C_U], norb, cptr_buffer[C_Ainv], lda, &czero, 
                             ptr_buffer[C_tempMat], lda_Binv, nw).wait();

      //cudaErrorCheck(SYCL::applyW_batched(hstream, delay_list_mw_ptr, delay_count, tempMat_mw_ptr, lda_Binv, nw),
      auto e =applyW_batched(lead_q,mw_delay_list.data(), delay_count, ptr_buffer[C_tempMat], lda_Binv, nw);

      e = syclBLAS::gemm_batched(lead_q,nontrans, nontrans, norb, delay_count, delay_count, &cone,
                                 cptr_buffer[C_V], norb, cptr_buffer[C_Binv], lda_Binv, &czero, 
                                 ptr_buffer[C_U], norb, nw,{e});

      syclBLAS::gemm_batched(lead_q, nontrans, nontrans, norb, norb, delay_count, &cminusone,
                             cptr_buffer[C_U], norb, cptr_buffer[C_tempMat], lda_Binv, &cone, 
                             ptr_buffer[C_Ainv], lda, nw,{e}).wait();
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
    if (on_host) //THIS IS CALLED, HOW TO SET THIS???
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
    engine_leader.guard_no_delay();

    for (This_t& engine : engines)
      engine.get_ref_psiMinv().updateFrom();
  }

  auto& getLAhandles()
  {
    return *m_queue;
  }
};
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_MATRIX_DELAYED_UPDATE_SYCL_H
