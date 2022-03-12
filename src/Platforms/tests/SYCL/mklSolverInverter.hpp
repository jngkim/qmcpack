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

#ifndef QMCPLUSPLUS_MKLSOLVERINVERTOR_H
#define QMCPLUSPLUS_MKLSOLVERINVERTOR_H

#include "OhmmsPETE/OhmmsVector.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "SYCL/SYCLruntime.hpp"
#include "SYCL/SYCLallocator.hpp"
#include "SYCL/syclBLAS.hpp"
#include "oneapi/mkl/lapack.hpp"

namespace qmcplusplus
{

namespace syclSolver=oneapi::mkl::lapack;

/** implements matrix inversion via cuSolverDN
 * @tparam T_FP high precision for matrix inversion, T_FP >= T
 */
template<typename T_FP>
class mklSolverInverter
{
  /// scratch memory for cusolverDN
  Matrix<T_FP, SYCLAllocator<T_FP>> Mat1_gpu;
  /// pivot array + info
  Vector<std::int64_t, SYCLHostAllocator<std::int64_t>> ipiv;
  /// workspace
  Vector<T_FP, SYCLAllocator<T_FP>> workspace;
  sycl::queue* m_queue=nullptr;
  std::int64_t getrf_ws=0;
  std::int64_t getri_ws=0;

  /** resize the internal storage
   * @param norb number of electrons/orbitals
   * @param delay, maximum delay 0<delay<=norb
   */
  inline void resize(int norb)
  {
    if(m_queue == nullptr) m_queue=get_default_queue();

    if (ipiv.size() != norb)
    {
      Mat1_gpu.resize(norb, norb);
      ipiv.resize(norb);
      getrf_ws=syclSolver::getrf_scratchpad_size<T_FP>(*m_queue,norb,norb,norb);
      getri_ws=syclSolver::getri_scratchpad_size<T_FP>(*m_queue,norb,norb);
      workspace.resize(std::max(getrf_ws,getri_ws));
    }
  }

public:

  /** compute the inverse of the transpose of matrix A and its determinant value in log
   * when T_FP and TMAT are the same
   * @tparam TREAL real type
   */
  template<typename TMAT, typename TREAL, typename = std::enable_if_t<std::is_same<TMAT, T_FP>::value>>
  std::enable_if_t<std::is_same<TMAT, T_FP>::value> invert_transpose(const Matrix<TMAT>& logdetT,
                                                                     Matrix<TMAT,OMPallocator<TMAT>>& Ainv_gpu,
                                                                     std::complex<TREAL>& log_value)
  {
    const int norb = logdetT.rows();
    resize(norb);

    //Mat1_gpu is used for transpose
    m_queue->memcpy(Mat1_gpu.data(),logdetT.data(),logdetT.size()*sizeof(TMAT)).wait();
    syclBLAS::transpose(*m_queue,Mat1_gpu.data(),norb,Mat1_gpu.cols(),Ainv_gpu.device_data(),norb,Ainv_gpu.cols()).wait();

    syclSolver::getrf(*m_queue,norb,norb,Ainv_gpu.device_data(),norb, ipiv.data(), workspace.data(), getrf_ws).wait();
    
    syclSolver::getri(*m_queue,norb,Ainv_gpu.device_data(),norb,ipiv.data(), workspace.data(), getri_ws).wait();
  }

  /** compute the inverse of the transpose of matrix A and its determinant value in log
   * when T_FP and TMAT are not the same
   * @tparam TREAL real type
   */
  template<typename TMAT, typename TREAL, typename = std::enable_if_t<!std::is_same<TMAT, T_FP>::value>>
  std::enable_if_t<!std::is_same<TMAT, T_FP>::value> invert_transpose(const Matrix<TMAT>& logdetT,
                                                                      Matrix<TMAT,OMPallocator<TMAT>>& Ainv_gpu,
                                                                      std::complex<TREAL>& log_value)
  {
    const int norb = logdetT.rows();
    resize(norb);

    m_queue->memcpy(Ainv_gpu.device_data(),logdetT.data(),logdetT.size()*sizeof(TMAT)).wait();
    //transpose
    auto e_t = syclBLAS::transpose(*m_queue,Ainv_gpu.device_data(),norb,Ainv_gpu.cols(),Mat1_gpu.data(),norb,Mat1_gpu.cols());

    //getrf (LU) -> getri (inverse)
    syclSolver::getrf(*m_queue,norb,norb,Mat1_gpu.data(),norb, ipiv.data(), workspace.data(), getrf_ws, {e_t}).wait();

    syclSolver::getri(*m_queue,norb,Mat1_gpu.data(),norb,ipiv.data(), workspace.data(), getri_ws).wait();

    syclBLAS::copy_n(*m_queue,Mat1_gpu.data(),Mat1_gpu.size(),Ainv_gpu.device_data()).wait();
  }
};
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_CUSOLVERINVERTOR_H
