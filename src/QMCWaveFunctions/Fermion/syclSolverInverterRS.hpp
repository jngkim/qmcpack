//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_SYCL_MKL_SOLVERINVERTOR_GETRS_H
#define QMCPLUSPLUS_SYCL_MKL_SOLVERINVERTOR_GETRS_H

#include "Containers/OhmmsPETE/OhmmsVector.h"
#include "Containers/OhmmsPETE/OhmmsMatrix.h"
#include "SYCL/SYCLallocator.hpp"
#include "SYCL/syclBLAS.hpp"
#include "SYCL/syclSolver.hpp"
#include "QMCWaveFunctions/detail/SYCL/sycl_determinant_helper.hpp"

namespace qmcplusplus
{
/** implements matrix inversion via cuSolverDN
 * @tparam T_FP high precision for matrix inversion, T_FP >= T
 */
template<typename T_FP>
class syclSolverInverterRS
{
  /// scratch memory for mklSolver
  Matrix<T_FP, SYCLAllocator<T_FP>> Mat1_gpu;
  Matrix<T_FP, SYCLAllocator<T_FP>> Mat2_gpu;
  /// workspace
  std::int64_t getrf_ws = 0;
  std::int64_t getrs_ws = 0;
  /// pivot array + info
  Vector<std::int64_t, SYCLAllocator<std::int64_t>> ipiv;
  Vector<T_FP, SYCLAllocator<T_FP>> getrf_scratch, getrs_scratch;

  /** resize the internal storage
   * @param norb number of electrons/orbitals
   * @param delay, maximum delay 0<delay<=norb
   */
  inline void resize(int norb, sycl::queue& m_queue)
  {
    if (Mat1_gpu.rows() != norb)
    {
      Mat1_gpu.resize(norb, norb);
      ipiv.resize(norb);
      getrf_ws = syclSolver::getrf_scratchpad_size<T_FP>(m_queue, norb, norb, norb);
      getrf_scratch.resize(getrf_ws);

      oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;
      getrs_ws = syclSolver::getrs_scratchpad_size<T_FP>(m_queue, trans, norb, norb, norb, norb);
      getrs_scratch.resize(getrs_ws);
    }
  }

public:
  /** compute the inverse of the transpose of matrix A and its determinant value in log
   * when T_FP and TMAT are the same
   * @tparam TREAL real type
   */
  template<typename TMAT, typename TREAL, typename = std::enable_if_t<std::is_same<TMAT, T_FP>::value>>
  std::enable_if_t<std::is_same<TMAT, T_FP>::value> invert_transpose(const Matrix<TMAT>& logdetT,
                                                                     Matrix<TMAT>& Ainv,
                                                                     Matrix<TMAT, SYCLAllocator<TMAT>>& Ainv_gpu,
                                                                     std::complex<TREAL>& log_value,
                                                                     sycl::queue& m_queue)
  {
    const int norb = logdetT.rows();
    resize(norb, m_queue);

    auto c_event = m_queue.memcpy(Mat1_gpu.data(), logdetT.data(), logdetT.size() * sizeof(TMAT));
    try
    {
      syclSolver::getrf(m_queue, norb, norb, Mat1_gpu.data(), norb, ipiv.data(), 
                        getrf_scratch.data(), getrf_ws, {c_event}).wait();
    }
    catch (sycl::exception const& ex)
    {
      std::cout << "\t\tCaught synchronous SYCL exception during getrf:\n"
                << ex.what() << "  status: " << ex.code() << std::endl;
      abort();
    }

    log_value = computeLogDet_sycl<TREAL>(m_queue, norb, Mat1_gpu.cols(), Mat1_gpu.data(), ipiv.data());
    auto s_event = make_identity_matrix_sycl(m_queue, norb, Ainv_gpu.data(), norb); 
    oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;
    auto i_event = syclSolver::getrs(m_queue, trans, norb, norb, Mat1_gpu.data(), norb, ipiv.data(), 
                                     Ainv_gpu.data(), norb, getrs_scratch.data(), getrf_ws, {s_event});

    m_queue.memcpy(Ainv.data(), Ainv_gpu.data(), Ainv.size() * sizeof(TMAT), {i_event}).wait();
  }

  /** compute the inverse of the transpose of matrix A and its determinant value in log
   * when T_FP and TMAT are not the same
   * @tparam TREAL real type
   */
  template<typename TMAT, typename TREAL, typename = std::enable_if_t<!std::is_same<TMAT, T_FP>::value>>
  std::enable_if_t<!std::is_same<TMAT, T_FP>::value> invert_transpose(const Matrix<TMAT>& logdetT,
                                                                      Matrix<TMAT>& Ainv,
                                                                      Matrix<TMAT, SYCLAllocator<TMAT>>& Ainv_gpu,
                                                                      std::complex<TREAL>& log_value,
                                                                      sycl::queue& m_queue)
  {
    const int norb = logdetT.rows();
    resize(norb, m_queue);
    Mat2_gpu.resize(norb, norb);

    //host -> device
    auto c_event_0 = m_queue.memcpy(Ainv_gpu.data(), logdetT.data(), logdetT.size() * sizeof(TMAT));
    //TMAT -> T_FP
    auto c_event_1 = syclBLAS::copy_n(m_queue, Ainv_gpu.data(), Ainv_gpu.size(), Mat1_gpu.data(), {c_event_0});
    try
    { //LU
      syclSolver::getrf(m_queue, norb, norb, Mat1_gpu.data(), norb, ipiv.data(), getrf_scratch.data(), getrf_ws,
                        {c_event_1}).wait();
    }
    catch (sycl::exception const& ex)
    {
      std::cout << "\t\tCaught synchronous SYCL exception during getrf:\n"
                << ex.what() << "  status: " << ex.code() << std::endl;
      abort();
    }

    auto i_event   = make_identity_matrix_sycl(m_queue, norb, Mat2_gpu.data(), norb); 
    log_value = computeLogDet_sycl<TREAL>(m_queue, norb, Mat1_gpu.cols(), Mat1_gpu.data(), ipiv.data());

    oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;
    auto rs_event = syclSolver::getrs(m_queue, trans, norb, norb, Mat1_gpu.data(), norb, ipiv.data(), 
                                      Mat2_gpu.data(), norb, getrs_scratch.data(), getrs_ws, {i_event});

    //T_FP -> TMAT
    auto c_event_2 = syclBLAS::copy_n(m_queue, Mat2_gpu.data(), Mat2_gpu.size(), Ainv_gpu.data(), {rs_event});

    //device -> host
    m_queue.memcpy(Ainv.data(), Ainv_gpu.data(), Ainv.size() * sizeof(TMAT), {c_event_2}).wait();
  }
};
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_CUSOLVERINVERTOR_H
