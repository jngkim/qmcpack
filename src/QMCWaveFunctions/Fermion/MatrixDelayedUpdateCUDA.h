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

#ifndef QMCPLUSPLUS_MATRIX_DELAYED_UPDATE_CUDA_H
#define QMCPLUSPLUS_MATRIX_DELAYED_UPDATE_CUDA_H

#include "OhmmsPETE/OhmmsVector.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "DualAllocatorAliases.hpp"
#include "QMCWaveFunctions/Fermion/DiracMatrix.h"
#include "Platforms/OMPTarget/ompBLAS.hpp"
#include "CUDA/CUDAruntime.hpp"
#include "CUDA/cuBLAS.hpp"
#include "CUDA/cuBLAS_missing_functions.hpp"
#include "QMCWaveFunctions/detail/CUDA/matrix_update_helper.hpp"
#include "DualAllocatorAliases.hpp"
#include "DiracMatrixComputeCUDA.hpp"
#include "WaveFunctionTypes.hpp"
#include "QueueAliases.hpp"
#include "AccelBLASAliases.hpp"

namespace qmcplusplus
{

/** implements dirac matrix delayed update using OpenMP offload and CUDA.
 * It is used as DET_ENGINE in DiracDeterminantBatched.
 * This is a 1 per walker class
 *
 * @tparam T base precision for most computation
 * @tparam T_FP high precision for matrix inversion, T_FP >= T
 */
template<typename VALUE, typename VALUE_FP>
class MatrixDelayedUpdateCUDA
{
public:
  using WFT           = WaveFunctionTypes<VALUE, VALUE_FP>;
  using Value         = typename WFT::Value;
  using Complex       = typename WFT::Complex;
  using FullPrecValue = typename WFT::FullPrecValue;
  using LogValue      = typename WFT::LogValue;
  using This_t        = MatrixDelayedUpdateCUDA<VALUE, VALUE_FP>;
  using DetInverter   = DiracMatrixComputeCUDA<FullPrecValue>;

  template<typename DT>
  using PinnedDualAllocator = PinnedDualAllocator<DT>;
  // Want to emphasize these because at least for cuda they can't be transferred async, which is bad.
  template<typename DT>
  using UnpinnedDualVector = Vector<DT, UnpinnedDualAllocator<DT>>;
  template<typename DT>
  using DualVector = Vector<DT, PinnedDualAllocator<DT>>;
  template<typename DT>
  using DualMatrix = Matrix<DT, PinnedDualAllocator<DT>>;
  template<typename DT>
  using DualVGLVector = VectorSoaContainer<DT, QMCTraits::DIM + 2, PinnedDualAllocator<DT>>;
  template<typename DT>
  using OffloadMWVGLArray = Array<DT, 3, OffloadPinnedAllocator<DT>>; // [VGL, walker, Orbs]
  template<typename DT>
  using OffloadMatrix = Matrix<DT, OffloadPinnedAllocator<DT>>;

  struct MultiWalkerResource
  {
    // CUDA stream, cublas handle object
    compute::Queue<PlatformKind::CUDA> queue;
    compute::BLAS<PlatformKind::CUDA> blas_handle;

    // constant array value VALUE(1)
    UnpinnedDualVector<Value> cone_vec;
    // constant array value VALUE(-1)
    UnpinnedDualVector<Value> cminusone_vec;
    // constant array value VALUE(0)
    UnpinnedDualVector<Value> czero_vec;
    // multi walker of grads for transfer needs.
    DualMatrix<Value> grads_value_v;
    // mw_updateRow pointer buffer
    Vector<char, PinnedDualAllocator<char>> updateRow_buffer_H2D;
    // mw_prepareInvRow pointer buffer
    Vector<char, PinnedDualAllocator<char>> prepare_inv_row_buffer_H2D;
    // mw_accept_rejectRow pointer buffer
    Vector<char, PinnedDualAllocator<char>> accept_rejectRow_buffer_H2D;
    // mw_updateInv pointer buffer
    Vector<char, PinnedDualAllocator<char>> updateInv_buffer_H2D;
    // mw_evalGrad pointer buffer
    Vector<char, PinnedDualAllocator<char>> evalGrad_buffer_H2D;
    /// scratch space for rank-1 update
    UnpinnedDualVector<Value> mw_temp;
    // scratch space for keeping one row of Ainv
    UnpinnedDualVector<Value> mw_rcopy;

    MultiWalkerResource() : blas_handle(queue.getNative()) {}

    void resize_fill_constant_arrays(size_t nw)
    {
      if (cone_vec.size() < nw)
      {
        // cone
        cone_vec.resize(nw);
        std::fill_n(cone_vec.data(), nw, Value(1));
        cone_vec.updateTo();
        // cminusone
        cminusone_vec.resize(nw);
        std::fill_n(cminusone_vec.data(), nw, Value(-1));
        cminusone_vec.updateTo();
        // czero
        czero_vec.resize(nw);
        std::fill_n(czero_vec.data(), nw, Value(0));
        czero_vec.updateTo();
      }
    }

    auto& getLAhandles() { return blas_handle; }
  };

private:
  /// scratch space for rank-1 update
  UnpinnedDualVector<Value> temp;
  /// row of up-to-date Ainv
  UnpinnedDualVector<Value> invRow;
  /** row id correspond to the up-to-date invRow. [0 norb), invRow is ready; -1, invRow is not valid.
   *  This id is set after calling getInvRow indicating invRow has been prepared for the invRow_id row
   *  ratioGrad checks if invRow_id is consistent. If not, invRow needs to be recomputed.
   *  acceptMove and completeUpdates mark invRow invalid by setting invRow_id to -1
   */
  int invRow_id;
  // scratch space for keeping one row of Ainv
  UnpinnedDualVector<Value> rcopy;

  template<typename DT>
  using DeviceMatrix = Matrix<DT, DeviceAllocator<DT>>;
  template<typename DT>
  using DeviceVector = Vector<DT, DeviceAllocator<DT>>;
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
  DeviceVector<int> delay_list_gpu;
  /// current number of delays, increase one for each acceptance, reset to 0 after updating Ainv
  int delay_count;

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
  static void mw_prepareInvRow(const RefVectorWithLeader<This_t>& engines,
                               MultiWalkerResource& mw_rsc,
                               const RefVector<DualMatrix<Value>>& psiMinv_refs,
                               const int rowchanged)
  {
    auto& engine_leader              = engines.getLeader();
    auto& blas_handle                = mw_rsc.blas_handle;
    auto& h_cublas                   = blas_handle.h_cublas;
    auto& queue                      = mw_rsc.queue;
    auto& cminusone_vec              = mw_rsc.cminusone_vec;
    auto& cone_vec                   = mw_rsc.cone_vec;
    auto& czero_vec                  = mw_rsc.czero_vec;
    auto& prepare_inv_row_buffer_H2D = mw_rsc.prepare_inv_row_buffer_H2D;
    const int norb                   = engine_leader.invRow.size();
    const int nw                     = engines.size();
    int& delay_count                 = engine_leader.delay_count;

    constexpr size_t num_ptrs_packed = 7; // it must match packing and unpacking
    prepare_inv_row_buffer_H2D.resize(sizeof(Value*) * num_ptrs_packed * nw);
    mw_rsc.resize_fill_constant_arrays(nw);

    const int lda_Binv = engine_leader.Binv_gpu.cols();
    Matrix<Value*> ptr_buffer(reinterpret_cast<Value**>(prepare_inv_row_buffer_H2D.data()), num_ptrs_packed, nw);
    for (int iw = 0; iw < nw; iw++)
    {
      This_t& engine             = engines[iw];
      DualMatrix<Value>& psiMinv = psiMinv_refs[iw];
      ptr_buffer[0][iw]          = psiMinv.device_data() + rowchanged * psiMinv.cols();
      ptr_buffer[1][iw]          = engine.invRow.device_data();
      ptr_buffer[2][iw]          = engine.U_gpu.data();
      ptr_buffer[3][iw]          = engine.p_gpu.data();
      ptr_buffer[4][iw]          = engine.Binv_gpu.data();
      ptr_buffer[5][iw]          = engine.Binv_gpu.data() + delay_count * lda_Binv;
      ptr_buffer[6][iw]          = engine.V_gpu.data();
    }

    queue.enqueueH2D(prepare_inv_row_buffer_H2D);

    Value** oldRow_mw_ptr = reinterpret_cast<Value**>(prepare_inv_row_buffer_H2D.device_data());
    Value** invRow_mw_ptr = reinterpret_cast<Value**>(prepare_inv_row_buffer_H2D.device_data() + sizeof(Value*) * nw);
    Value** U_mw_ptr    = reinterpret_cast<Value**>(prepare_inv_row_buffer_H2D.device_data() + sizeof(Value*) * nw * 2);
    Value** p_mw_ptr    = reinterpret_cast<Value**>(prepare_inv_row_buffer_H2D.device_data() + sizeof(Value*) * nw * 3);
    Value** Binv_mw_ptr = reinterpret_cast<Value**>(prepare_inv_row_buffer_H2D.device_data() + sizeof(Value*) * nw * 4);
    Value** BinvRow_mw_ptr =
        reinterpret_cast<Value**>(prepare_inv_row_buffer_H2D.device_data() + sizeof(Value*) * nw * 5);
    Value** V_mw_ptr = reinterpret_cast<Value**>(prepare_inv_row_buffer_H2D.device_data() + sizeof(Value*) * nw * 6);

    // save Ainv[rowchanged] to invRow
    //std::copy_n(Ainv[rowchanged], norb, invRow.data());
    cudaErrorCheck(cuBLAS_MFs::copy_batched(queue.getNative(), norb, oldRow_mw_ptr, 1, invRow_mw_ptr, 1, nw),
                   "cuBLAS_MFs::copy_batched failed!");
    // multiply V (NxK) Binv(KxK) U(KxN) invRow right to the left
    //BLAS::gemv('T', norb, delay_count, cone, U_gpu.data(), norb, invRow.data(), 1, czero, p_gpu.data(), 1);
    //BLAS::gemv('N', delay_count, delay_count, -cone, Binv.data(), lda_Binv, p.data(), 1, czero, Binv[delay_count], 1);
    //BLAS::gemv('N', norb, delay_count, cone, V.data(), norb, Binv[delay_count], 1, cone, invRow.data(), 1);
    cudaErrorCheck(cuBLAS_MFs::gemv_batched(queue.getNative(), 'T', norb, delay_count, cone_vec.device_data(), U_mw_ptr,
                                            norb, invRow_mw_ptr, 1, czero_vec.device_data(), p_mw_ptr, 1, nw),
                   "cuBLAS_MFs::gemv_batched failed!");
    cudaErrorCheck(cuBLAS_MFs::gemv_batched(queue.getNative(), 'N', delay_count, delay_count,
                                            cminusone_vec.device_data(), Binv_mw_ptr, lda_Binv, p_mw_ptr, 1,
                                            czero_vec.device_data(), BinvRow_mw_ptr, 1, nw),
                   "cuBLAS_MFs::gemv_batched failed!");
    cudaErrorCheck(cuBLAS_MFs::gemv_batched(queue.getNative(), 'N', norb, delay_count, cone_vec.device_data(), V_mw_ptr,
                                            norb, BinvRow_mw_ptr, 1, cone_vec.device_data(), invRow_mw_ptr, 1, nw),
                   "cuBLAS_MFs::gemv_batched failed!");
    // mark row prepared
    engine_leader.invRow_id = rowchanged;
  }

  /** Do complete row updates
   *  many of these const arguments provide pointers or references
   *  somewhere in here is an update that doesn't get where it belongs resulting in a 0
   *  gradient later.
   *  Sad example of OpenMP target code that is far from clear and a poor substitute for a
   *  clear CPU reference implementation.
   *
   *  \param[in] engines
   *  \param[in] rowchanged
   *  \param[in] psiM_g_list        device ptrs
   *  \param[in] psiM_l_list        device ptrs
   *  \param[in] isAccepted         bool but wait some lists are also filtered
   *  \param[in] phi_vgl_v          multiple walker orbital VGL
   *  \param[inout] ratios
   */
  static void mw_updateRow(const RefVectorWithLeader<This_t>& engines,
                           MultiWalkerResource& mw_rsc,
                           const RefVector<DualMatrix<Value>>& psiMinv_refs,
                           const int rowchanged,
                           const std::vector<Value*>& psiM_g_list,
                           const std::vector<Value*>& psiM_l_list,
                           const std::vector<bool>& isAccepted,
                           const OffloadMWVGLArray<Value>& phi_vgl_v,
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

    auto& queue                 = mw_rsc.queue;
    auto& updateRow_buffer_H2D  = mw_rsc.updateRow_buffer_H2D;
    auto& mw_temp               = mw_rsc.mw_temp;
    auto& mw_rcopy              = mw_rsc.mw_rcopy;
    auto& cone_vec              = mw_rsc.cone_vec;
    auto& czero_vec             = mw_rsc.czero_vec;
    const int norb              = engine_leader.invRow.size();
    const int lda               = psiMinv_refs[0].get().cols();
    const int nw                = engines.size();
    const size_t phi_vgl_stride = nw * norb;
    mw_temp.resize(norb * n_accepted);
    mw_rcopy.resize(norb * n_accepted);

    constexpr size_t num_ptrs_packed = 6; // it must match packing and unpacking
    updateRow_buffer_H2D.resize((sizeof(Value*) * num_ptrs_packed + sizeof(Value)) * n_accepted);

    // to handle T** of Ainv, psi_v, temp, rcopy
    Matrix<Value*> ptr_buffer(reinterpret_cast<Value**>(updateRow_buffer_H2D.data()), num_ptrs_packed, n_accepted);
    Value* c_ratio_inv =
        reinterpret_cast<Value*>(updateRow_buffer_H2D.data() + sizeof(Value*) * num_ptrs_packed * n_accepted);
    for (int iw = 0, count = 0; iw < isAccepted.size(); iw++)
      if (isAccepted[iw])
      {
        ptr_buffer[0][count] = psiMinv_refs[iw].get().device_data();
        ptr_buffer[1][count] = const_cast<Value*>(phi_vgl_v.device_data_at(0, iw, 0));
        ptr_buffer[2][count] = mw_temp.device_data() + norb * count;
        ptr_buffer[3][count] = mw_rcopy.device_data() + norb * count;
        ptr_buffer[4][count] = psiM_g_list[count];
        ptr_buffer[5][count] = psiM_l_list[count];

        c_ratio_inv[count] = Value(-1) / ratios[iw];
        count++;
      }

    // update the inverse matrix
    mw_rsc.resize_fill_constant_arrays(n_accepted);

    queue.enqueueH2D(updateRow_buffer_H2D);

    {
      Value** Ainv_mw_ptr = reinterpret_cast<Value**>(updateRow_buffer_H2D.device_data());
      Value** phiVGL_mw_ptr =
          reinterpret_cast<Value**>(updateRow_buffer_H2D.device_data() + sizeof(Value*) * n_accepted);
      Value** temp_mw_ptr =
          reinterpret_cast<Value**>(updateRow_buffer_H2D.device_data() + sizeof(Value*) * n_accepted * 2);
      Value** rcopy_mw_ptr =
          reinterpret_cast<Value**>(updateRow_buffer_H2D.device_data() + sizeof(Value*) * n_accepted * 3);
      Value** dpsiM_mw_out =
          reinterpret_cast<Value**>(updateRow_buffer_H2D.device_data() + sizeof(Value*) * n_accepted * 4);
      Value** d2psiM_mw_out =
          reinterpret_cast<Value**>(updateRow_buffer_H2D.device_data() + sizeof(Value*) * n_accepted * 5);
      Value* ratio_inv_mw =
          reinterpret_cast<Value*>(updateRow_buffer_H2D.device_data() + sizeof(Value*) * n_accepted * 6);


      // invoke the Fahy's variant of Sherman-Morrison update.
      cudaErrorCheck(cuBLAS_MFs::gemv_batched(queue.getNative(), 'T', norb, norb, cone_vec.device_data(), Ainv_mw_ptr,
                                              lda, phiVGL_mw_ptr, 1, czero_vec.device_data(), temp_mw_ptr, 1,
                                              n_accepted),
                     "cuBLAS_MFs::gemv_batched failed!");

      cudaErrorCheck(CUDA::copyAinvRow_saveGL_cuda(queue.getNative(), rowchanged, norb, Ainv_mw_ptr, lda, temp_mw_ptr,
                                                   rcopy_mw_ptr, phiVGL_mw_ptr, phi_vgl_stride, dpsiM_mw_out,
                                                   d2psiM_mw_out, n_accepted),
                     "CUDA::copyAinvRow_saveGL_cuda failed!");


      cudaErrorCheck(cuBLAS_MFs::ger_batched(queue.getNative(), norb, norb, ratio_inv_mw, rcopy_mw_ptr, 1, temp_mw_ptr,
                                             1, Ainv_mw_ptr, lda, n_accepted),
                     "cuBLAS_MFs::ger_batched failed!");
    }
  }

public:
  /// default constructor
  MatrixDelayedUpdateCUDA() : invRow_id(-1), delay_count(0) {}

  MatrixDelayedUpdateCUDA(const MatrixDelayedUpdateCUDA&) = delete;

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
  }

  // prepare invRow and compute the old gradients.
  template<typename GT>
  static void mw_evalGrad(const RefVectorWithLeader<This_t>& engines,
                          MultiWalkerResource& mw_rsc,
                          const RefVector<DualMatrix<Value>>& psiMinv_refs,
                          const std::vector<const Value*>& dpsiM_row_list,
                          const int rowchanged,
                          std::vector<GT>& grad_now)
  {
    auto& engine_leader = engines.getLeader();
    if (!engine_leader.isSM1())
      mw_prepareInvRow(engines, mw_rsc, psiMinv_refs, rowchanged);

    auto& queue               = mw_rsc.queue;
    auto& evalGrad_buffer_H2D = mw_rsc.evalGrad_buffer_H2D;
    auto& grads_value_v       = mw_rsc.grads_value_v;

    const int nw                     = engines.size();
    constexpr size_t num_ptrs_packed = 2; // it must match packing and unpacking
    evalGrad_buffer_H2D.resize(sizeof(Value*) * num_ptrs_packed * nw);
    Matrix<const Value*> ptr_buffer(reinterpret_cast<const Value**>(evalGrad_buffer_H2D.data()), num_ptrs_packed, nw);
    for (int iw = 0; iw < nw; iw++)
    {
      if (engine_leader.isSM1())
      {
        DualMatrix<Value>& psiMinv = psiMinv_refs[iw];
        ptr_buffer[0][iw]          = psiMinv.device_data() + rowchanged * psiMinv.cols();
      }
      else
        ptr_buffer[0][iw] = engines[iw].invRow.device_data();
      ptr_buffer[1][iw] = dpsiM_row_list[iw];
    }

    queue.enqueueH2D(evalGrad_buffer_H2D);

    if (grads_value_v.rows() != nw || grads_value_v.cols() != GT::Size)
      grads_value_v.resize(nw, GT::Size);

    const Value** invRow_ptr    = reinterpret_cast<const Value**>(evalGrad_buffer_H2D.device_data());
    const Value** dpsiM_row_ptr = reinterpret_cast<const Value**>(evalGrad_buffer_H2D.device_data()) + nw;

    cudaErrorCheck(CUDA::calcGradients_cuda(queue.getNative(), engine_leader.invRow.size(), invRow_ptr, dpsiM_row_ptr,
                                            grads_value_v.device_data(), nw),
                   "CUDA::calcGradients_cuda failed!");
    queue.enqueueD2H(grads_value_v);
    queue.sync();

    for (int iw = 0; iw < nw; iw++)
      grad_now[iw] = {grads_value_v[iw][0], grads_value_v[iw][1], grads_value_v[iw][2]};
  }

  template<typename GT>
  static void mw_evalGradWithSpin(const RefVectorWithLeader<This_t>& engines,
                                  MultiWalkerResource& mw_rsc,
                                  const RefVector<DualMatrix<Value>>& psiMinv_refs,
                                  const std::vector<const Value*>& dpsiM_row_list,
                                  OffloadMatrix<Complex>& mw_dspin,
                                  const int rowchanged,
                                  std::vector<GT>& grad_now,
                                  std::vector<Complex>& spingrad_now)
  {
    throw std::runtime_error("MatrixDelayedUpdateCUDA needs implementation of mw_evalGradWithSpin");
  }

  /** Update the "local" psiMinv_ on the device.
   *  Side Effect Transfers:
   *  * psiMinv_ is transferred back to host since single calls from QMCHamitonian and others
   *  * expect it to be.
   *
   *  Forced to use OpenMP target since resources are banned for single walker functions APIs
   *  and the acquireRelease pattern for a single DDB was removed by #3324
   */
  template<typename VVT>
  void updateRow(DualMatrix<Value>& Ainv, int rowchanged, const VVT& phiV, FullPrecValue c_ratio_in)
  {
    guard_no_delay();
    // update the inverse matrix
    constexpr Value cone(1), czero(0);
    const int norb = Ainv.rows();
    const int lda  = Ainv.cols();
    temp.resize(norb);
    rcopy.resize(norb);
    // invoke the Fahy's variant of Sherman-Morrison update.
    int dummy_handle      = 0;
    const Value* phiV_ptr = phiV.data();
    Value* Ainv_ptr       = Ainv.data();
    Value* temp_ptr       = temp.data();
    Value* rcopy_ptr      = rcopy.data();
    // psiMinv_ must be update-to-date on both host and device
    PRAGMA_OFFLOAD("omp target data map(always, tofrom: Ainv_ptr[:Ainv.size()]) \
                    use_device_ptr(phiV_ptr, Ainv_ptr, temp_ptr, rcopy_ptr)")
    {
      int success = ompBLAS::gemv(dummy_handle, 'T', norb, norb, cone, Ainv_ptr, lda, phiV_ptr, 1, czero, temp_ptr, 1);
      if (success != 0)
        throw std::runtime_error("ompBLAS::gemv failed.");

      PRAGMA_OFFLOAD("omp target parallel for simd is_device_ptr(Ainv_ptr, temp_ptr, rcopy_ptr)")
      for (int i = 0; i < norb; i++)
      {
        rcopy_ptr[i] = Ainv_ptr[rowchanged * lda + i];
        if (i == 0)
          temp_ptr[rowchanged] -= cone;
      }

      success = ompBLAS::ger(dummy_handle, norb, norb, static_cast<Value>(FullPrecValue(-1) / c_ratio_in), rcopy_ptr, 1,
                             temp_ptr, 1, Ainv_ptr, lda);
      if (success != 0)
        throw std::runtime_error("ompBLAS::ger failed.");
    }
  }

  /** Accept or Reject row updates
   *  many of these const arguments provide pointers or references
   *  to objects that do get modified.
   *  \param[in] engines
   *  \param[in] rowchanged
   *  \param[in] psiM_g_list
   *  \param[in] psiM_l_list
   *  \param[in] isAccepted
   *  \param[in] phi_vgl_v          multiple walker orbital VGL
   *  \param[inout] ratios
   */
  static void mw_accept_rejectRow(const RefVectorWithLeader<This_t>& engines,
                                  MultiWalkerResource& mw_rsc,
                                  const RefVector<DualMatrix<Value>>& psiMinv_refs,
                                  const int rowchanged,
                                  const std::vector<Value*>& psiM_g_list,
                                  const std::vector<Value*>& psiM_l_list,
                                  const std::vector<bool>& isAccepted,
                                  const OffloadMWVGLArray<Value>& phi_vgl_v,
                                  const std::vector<Value>& ratios)
  {
    auto& engine_leader = engines.getLeader();
    // invRow consumed, mark invRow_id unset
    engine_leader.invRow_id = -1;

    if (engine_leader.isSM1())
    {
      mw_updateRow(engines, mw_rsc, psiMinv_refs, rowchanged, psiM_g_list, psiM_l_list, isAccepted, phi_vgl_v, ratios);
      return;
    }

    auto& queue                       = mw_rsc.queue;
    auto& cminusone_vec               = mw_rsc.cminusone_vec;
    auto& cone_vec                    = mw_rsc.cone_vec;
    auto& czero_vec                   = mw_rsc.czero_vec;
    auto& accept_rejectRow_buffer_H2D = mw_rsc.accept_rejectRow_buffer_H2D;
    int& delay_count                  = engine_leader.delay_count;
    const int lda_Binv                = engine_leader.Binv_gpu.cols();
    const int norb                    = engine_leader.invRow.size();
    const int nw                      = engines.size();
    const int n_accepted              = psiM_g_list.size();
    const size_t phi_vgl_stride       = nw * norb;

    constexpr size_t num_ptrs_packed = 12; // it must match packing and unpacking
    accept_rejectRow_buffer_H2D.resize((sizeof(Value*) * num_ptrs_packed + sizeof(Value)) * nw);
    mw_rsc.resize_fill_constant_arrays(nw);

    Matrix<Value*> ptr_buffer(reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.data()), num_ptrs_packed, nw);
    Value* c_ratio_inv =
        reinterpret_cast<Value*>(accept_rejectRow_buffer_H2D.data() + sizeof(Value*) * num_ptrs_packed * nw);
    for (int iw = 0, count_accepted = 0, count_rejected = 0; iw < nw; iw++)
    {
      DualMatrix<Value>& psiMinv = psiMinv_refs[iw];
      const int lda              = psiMinv.cols();
      This_t& engine             = engines[iw];
      if (isAccepted[iw])
      {
        ptr_buffer[0][count_accepted]  = psiMinv.device_data() + lda * rowchanged;
        ptr_buffer[1][count_accepted]  = engine.V_gpu.data();
        ptr_buffer[2][count_accepted]  = engine.U_gpu.data() + norb * delay_count;
        ptr_buffer[3][count_accepted]  = engine.p_gpu.data();
        ptr_buffer[4][count_accepted]  = engine.Binv_gpu.data();
        ptr_buffer[5][count_accepted]  = engine.Binv_gpu.data() + delay_count * lda_Binv;
        ptr_buffer[6][count_accepted]  = engine.Binv_gpu.data() + delay_count;
        ptr_buffer[7][count_accepted]  = reinterpret_cast<Value*>(engine.delay_list_gpu.data());
        ptr_buffer[8][count_accepted]  = engine.V_gpu.data() + norb * delay_count;
        ptr_buffer[9][count_accepted]  = const_cast<Value*>(phi_vgl_v.device_data_at(0, iw, 0));
        ptr_buffer[10][count_accepted] = psiM_g_list[count_accepted];
        ptr_buffer[11][count_accepted] = psiM_l_list[count_accepted];
        c_ratio_inv[count_accepted]    = Value(1) / ratios[iw];
        count_accepted++;
      }
      else
      {
        ptr_buffer[0][n_accepted + count_rejected] = psiMinv.device_data() + lda * rowchanged;
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

    queue.enqueueH2D(accept_rejectRow_buffer_H2D);

    Value** invRow_mw_ptr = reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.device_data());
    Value** V_mw_ptr      = reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw);
    Value** U_row_mw_ptr =
        reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 2);
    Value** p_mw_ptr = reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 3);
    Value** Binv_mw_ptr =
        reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 4);
    Value** BinvRow_mw_ptr =
        reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 5);
    Value** BinvCol_mw_ptr =
        reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 6);
    int** delay_list_mw_ptr =
        reinterpret_cast<int**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 7);
    Value** V_row_mw_ptr =
        reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 8);
    Value** phiVGL_mw_ptr =
        reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 9);
    Value** dpsiM_mw_out =
        reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 10);
    Value** d2psiM_mw_out =
        reinterpret_cast<Value**>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 11);
    Value* ratio_inv_mw_ptr =
        reinterpret_cast<Value*>(accept_rejectRow_buffer_H2D.device_data() + sizeof(Value*) * nw * 12);

    //std::copy_n(Ainv[rowchanged], norb, V[delay_count]);
    cudaErrorCheck(cuBLAS_MFs::copy_batched(queue.getNative(), norb, invRow_mw_ptr, 1, V_row_mw_ptr, 1, nw),
                   "cuBLAS_MFs::copy_batched failed!");
    // handle accepted walkers
    // the new Binv is [[X Y] [Z sigma]]
    //BLAS::gemv('T', norb, delay_count + 1, cminusone, V.data(), norb, psiV.data(), 1, czero, p.data(), 1);
    cudaErrorCheck(cuBLAS_MFs::gemv_batched(queue.getNative(), 'T', norb, delay_count, cminusone_vec.device_data(),
                                            V_mw_ptr, norb, phiVGL_mw_ptr, 1, czero_vec.device_data(), p_mw_ptr, 1,
                                            n_accepted),
                   "cuBLAS_MFs::gemv_batched failed!");
    // Y
    //BLAS::gemv('T', delay_count, delay_count, sigma, Binv.data(), lda_Binv, p.data(), 1, czero, Binv.data() + delay_count,
    //           lda_Binv);
    cudaErrorCheck(cuBLAS_MFs::gemv_batched(queue.getNative(), 'T', delay_count, delay_count, ratio_inv_mw_ptr,
                                            Binv_mw_ptr, lda_Binv, p_mw_ptr, 1, czero_vec.device_data(), BinvCol_mw_ptr,
                                            lda_Binv, n_accepted),
                   "cuBLAS_MFs::gemv_batched failed!");
    // X
    //BLAS::ger(delay_count, delay_count, cone, Binv[delay_count], 1, Binv.data() + delay_count, lda_Binv,
    //          Binv.data(), lda_Binv);
    cudaErrorCheck(cuBLAS_MFs::ger_batched(queue.getNative(), delay_count, delay_count, cone_vec.device_data(),
                                           BinvRow_mw_ptr, 1, BinvCol_mw_ptr, lda_Binv, Binv_mw_ptr, lda_Binv,
                                           n_accepted),
                   "cuBLAS_MFs::ger_batched failed!");
    // sigma and Z
    cudaErrorCheck(CUDA::add_delay_list_save_sigma_VGL_batched(queue.getNative(), delay_list_mw_ptr, rowchanged,
                                                               delay_count, Binv_mw_ptr, lda_Binv, ratio_inv_mw_ptr,
                                                               phiVGL_mw_ptr, phi_vgl_stride, U_row_mw_ptr,
                                                               dpsiM_mw_out, d2psiM_mw_out, norb, n_accepted, nw),
                   "CUDA::add_delay_list_save_y_VGL_batched failed!");
    delay_count++;
    // update Ainv when maximal delay is reached
    if (delay_count == lda_Binv)
      mw_updateInvMat(engines, mw_rsc, psiMinv_refs);
  }

  /** update the full Ainv and reset delay_count
   * @param Ainv inverse matrix
   */
  static void mw_updateInvMat(const RefVectorWithLeader<This_t>& engines,
                              MultiWalkerResource& mw_rsc,
                              const RefVector<DualMatrix<Value>>& psiMinv_refs)
  {
    auto& engine_leader = engines.getLeader();
    int& delay_count    = engine_leader.delay_count;
    if (delay_count == 0)
      return;
    // update the inverse matrix
    auto& queue                = mw_rsc.queue;
    auto& h_cublas             = mw_rsc.blas_handle.h_cublas;
    auto& updateInv_buffer_H2D = mw_rsc.updateInv_buffer_H2D;
    const int norb             = engine_leader.invRow.size();
    const int lda              = psiMinv_refs[0].get().cols();
    const int nw               = engines.size();

    constexpr size_t num_ptrs_packed = 6; // it must match packing and unpacking
    updateInv_buffer_H2D.resize(sizeof(Value*) * num_ptrs_packed * nw);
    mw_rsc.resize_fill_constant_arrays(nw);

    Matrix<Value*> ptr_buffer(reinterpret_cast<Value**>(updateInv_buffer_H2D.data()), num_ptrs_packed, nw);
    for (int iw = 0; iw < nw; iw++)
    {
      This_t& engine    = engines[iw];
      ptr_buffer[0][iw] = engine.U_gpu.data();
      ptr_buffer[1][iw] = psiMinv_refs[iw].get().device_data();
      ptr_buffer[2][iw] = engine.tempMat_gpu.data();
      ptr_buffer[3][iw] = reinterpret_cast<Value*>(engine.delay_list_gpu.data());
      ptr_buffer[4][iw] = engine.V_gpu.data();
      ptr_buffer[5][iw] = engine.Binv_gpu.data();
    }

    queue.enqueueH2D(updateInv_buffer_H2D);

    Value** U_mw_ptr        = reinterpret_cast<Value**>(updateInv_buffer_H2D.device_data());
    Value** Ainv_mw_ptr     = reinterpret_cast<Value**>(updateInv_buffer_H2D.device_data() + sizeof(Value*) * nw);
    Value** tempMat_mw_ptr  = reinterpret_cast<Value**>(updateInv_buffer_H2D.device_data() + sizeof(Value*) * nw * 2);
    int** delay_list_mw_ptr = reinterpret_cast<int**>(updateInv_buffer_H2D.device_data() + sizeof(Value*) * nw * 3);
    Value** V_mw_ptr        = reinterpret_cast<Value**>(updateInv_buffer_H2D.device_data() + sizeof(Value*) * nw * 4);
    Value** Binv_mw_ptr     = reinterpret_cast<Value**>(updateInv_buffer_H2D.device_data() + sizeof(Value*) * nw * 5);

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
      const int lda_Binv = engine_leader.Binv_gpu.cols();
      constexpr Value cone(1), czero(0), cminusone(-1);
      cublasErrorCheck(cuBLAS::gemm_batched(h_cublas, CUBLAS_OP_T, CUBLAS_OP_N, delay_count, norb, norb, &cone,
                                            U_mw_ptr, norb, Ainv_mw_ptr, lda, &czero, tempMat_mw_ptr, lda_Binv, nw),
                       "cuBLAS::gemm_batched failed!");
      cudaErrorCheck(CUDA::applyW_batched(queue.getNative(), delay_list_mw_ptr, delay_count, tempMat_mw_ptr, lda_Binv,
                                          nw),
                     "CUDA::applyW_batched failed!");
      cublasErrorCheck(cuBLAS::gemm_batched(h_cublas, CUBLAS_OP_N, CUBLAS_OP_N, norb, delay_count, delay_count, &cone,
                                            V_mw_ptr, norb, Binv_mw_ptr, lda_Binv, &czero, U_mw_ptr, norb, nw),
                       "cuBLAS::gemm_batched failed!");
      cublasErrorCheck(cuBLAS::gemm_batched(h_cublas, CUBLAS_OP_N, CUBLAS_OP_N, norb, norb, delay_count, &cminusone,
                                            U_mw_ptr, norb, tempMat_mw_ptr, lda_Binv, &cone, Ainv_mw_ptr, lda, nw),
                       "cuBLAS::gemm_batched failed!");
    }
    delay_count = 0;
  }

  /*
  inline void print_Ainv(const RefVector<This_t>& engines)
  {
    for (This_t& engine : engines)
    {
      std::cout << "debug Ainv host  " << engine.get_psiMinv()[0][0] << " " << engine.get_psiMinv()[0][32] << " "
                << engine.get_psiMinv()[1][0] << " " << engine.get_psiMinv()[1][32] << std::endl;
      auto* temp_ptr = engine.get_psiMinv().data();
      PRAGMA_OFFLOAD("omp target update from(temp_ptr[:psiMinv_.size()])")
      std::cout << "debug Ainv devi";
      for (int row = 0; row < psiMinv_.rows(); row++)
      {
        for (int col = 0; col < psiMinv_.cols(); col++)
          std::cout << " " << row << " " << col << " " << engine.get_psiMinv()[row][col];
        std::cout << std::endl;
      }
    }
  }
*/

  /** return invRow host or device pointers based on on_host request
   * prepare invRow if not already.
   */
  static std::vector<const Value*> mw_getInvRow(const RefVectorWithLeader<This_t>& engines,
                                                MultiWalkerResource& mw_rsc,
                                                const RefVector<DualMatrix<Value>>& psiMinv_refs,
                                                const int row_id,
                                                bool on_host)
  {
    auto& engine_leader = engines.getLeader();
    auto& queue         = mw_rsc.queue;
    if (engine_leader.isSM1())
      queue.sync();
    else if (engine_leader.invRow_id != row_id)
    {
      // this can be skipped if mw_evalGrad gets called already.
      mw_prepareInvRow(engines, mw_rsc, psiMinv_refs, row_id);
      queue.sync();
    }

    std::vector<const Value*> row_ptr_list;
    row_ptr_list.reserve(psiMinv_refs.size());
    if (on_host)
    {
      // copy values to host and return host pointer
      if (engine_leader.isSM1())
        for (DualMatrix<Value>& psiMinv : psiMinv_refs)
        {
          const size_t ncols = psiMinv.cols();
          psiMinv.updateFrom(ncols, row_id * ncols);
          row_ptr_list.push_back(psiMinv.data() + row_id * ncols);
        }
      else
        for (This_t& engine : engines)
        {
          engine.invRow.updateFrom();
          row_ptr_list.push_back(engine.invRow.data());
        }
    }
    else
    {
      // return device pointer
      if (engine_leader.isSM1())
        for (DualMatrix<Value>& psiMinv : psiMinv_refs)
          row_ptr_list.push_back(psiMinv.device_data() + row_id * psiMinv.cols());
      else
        for (This_t& engine : engines)
          row_ptr_list.push_back(engine.invRow.device_data());
    }
    return row_ptr_list;
  }

  /// transfer Ainv to the host
  static void mw_transferAinv_D2H(const RefVectorWithLeader<This_t>& engines,
                                  MultiWalkerResource& mw_rsc,
                                  const RefVector<DualMatrix<Value>>& psiMinv_refs)
  {
    auto& engine_leader = engines.getLeader();
    auto& queue         = mw_rsc.queue;
    engine_leader.guard_no_delay();

    for (DualMatrix<Value>& psiMinv : psiMinv_refs)
      queue.enqueueD2H(psiMinv);
    queue.sync();
  }

  /** transfer psiM_vgl to the host. psiM_vgl has 5 rows, V(1) G(3) L(1)
   * @param engine_leader for accessing shared resource
   * @param psiM_vgl_list list of psiM_vgl
   * @param row_begin first row to copy
   * @param row_size the number of rows to be copied
   */
  static void mw_transferVGL_D2H(This_t& engine_leader,
                                 MultiWalkerResource& mw_rsc,
                                 const RefVector<DualVGLVector<Value>>& psiM_vgl_list,
                                 size_t row_begin,
                                 size_t row_size)
  {
    auto& queue = mw_rsc.queue;
    for (DualVGLVector<Value>& psiM_vgl : psiM_vgl_list)
    {
      const size_t stride = psiM_vgl.capacity();
      queue.enqueueD2H(psiM_vgl, row_size * stride, row_begin * stride);
    }
    queue.sync();
  }

  /** transfer psiM_vgl to the device. psiM_vgl has 5 rows, V(1) G(3) L(1)
   * @param engine_leader for accessing shared resource
   * @param psiM_vgl_list list of psiM_vgl
   * @param row_begin first row to copy
   * @param row_size the number of rows to be copied
   */
  static void mw_transferVGL_H2D(This_t& engine_leader,
                                 MultiWalkerResource& mw_rsc,
                                 const RefVector<DualVGLVector<Value>>& psiM_vgl_list,
                                 size_t row_begin,
                                 size_t row_size)
  {
    auto& queue = mw_rsc.queue;
    for (DualVGLVector<Value>& psiM_vgl : psiM_vgl_list)
    {
      const size_t stride = psiM_vgl.capacity();
      queue.enqueueH2D(psiM_vgl, row_size * stride, row_begin * stride);
    }
    queue.sync();
  }
};
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_MATRIX_DELAYED_UPDATE_CUDA_H
