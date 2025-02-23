//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_DIRAC_MATRIX_H
#define QMCPLUSPLUS_DIRAC_MATRIX_H

#include "CPU/BLAS.hpp"
#include "CPU/BlasThreadingEnv.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "type_traits/scalar_traits.h"
#include "Message/OpenMP.h"
#include "CPU/SIMD/simd.hpp"

namespace qmcplusplus
{

template<typename T, typename T_FP>
inline void computeLogDet(const T* restrict diag, int n, const lapack_int* restrict pivot, std::complex<T_FP>& logdet)
{
  logdet = std::complex<T_FP>();
  for (size_t i = 0; i < n; i++)
    logdet += std::log(std::complex<T_FP>((pivot[i] == i + 1) ? diag[i] : -diag[i]));
}

/** helper class to compute matrix inversion and the log value of determinant
 * @tparam T_FP the datatype used in the actual computation of matrix inversion
 */
template<typename T_FP>
class DiracMatrix
{
  typedef typename scalar_traits<T_FP>::real_type real_type_fp;
  aligned_vector<T_FP> m_work;
  aligned_vector<lapack_int> m_pivot;
  int Lwork;
  /// scratch space used for mixed precision
  Matrix<T_FP> psiM_fp;
  /// LU diagonal elements
  aligned_vector<T_FP> LU_diag;

  /// reset internal work space
  inline void reset(T_FP* invMat_ptr, const int lda)
  {
    m_pivot.resize(lda);
    Lwork = -1;
    T_FP tmp;
    real_type_fp lw;
    int status=LAPACK::getri(lda, invMat_ptr, lda, m_pivot.data(), &tmp, Lwork);
    if (status != 0)
    {
      std::ostringstream msg;
      msg << "Xgetri failed with error " << status << std::endl;
      throw std::runtime_error(msg.str());
    }

    convert(tmp, lw);
    Lwork = static_cast<int>(lw);
    m_work.resize(Lwork);
    LU_diag.resize(lda);
  }

  /** compute the inverse of invMat (in place) and the log value of determinant
   * @tparam TREAL real type
   * @param n invMat is n x n matrix
   * @param lda the first dimension of invMat
   * @param LogDet log determinant value of invMat before inversion
   */
  template<typename TREAL>
  inline void computeInvertAndLog(T_FP* invMat, const int n, const int lda, std::complex<TREAL>& LogDet)
  {
    BlasThreadingEnv knob(getNextLevelNumThreads());
    if (Lwork < lda)
      reset(invMat, lda);
    int status=LAPACK::getrf(n, n, invMat, lda, m_pivot.data());
    if (status != 0)
    {
      std::ostringstream msg;
      msg << "Xgetrf failed with error " << status << std::endl;
      throw std::runtime_error(msg.str());
    }
    for (int i = 0; i < n; i++)
      LU_diag[i] = invMat[i * lda + i];
    computeLogDet(LU_diag.data(), n, m_pivot.data(), LogDet);
    status=LAPACK::getri(n, invMat, lda, m_pivot.data(), m_work.data(), Lwork);
    if (status != 0)
    {
      std::ostringstream msg;
      msg << "Xgetri failed with error " << status << std::endl;
      throw std::runtime_error(msg.str());
    }
  }

public:
  DiracMatrix() : Lwork(0) {}

  /** compute the inverse of the transpose of matrix A and its determinant value in log
   * when T_FP and TMAT are the same
   * @tparam TMAT matrix value type
   * @tparam TREAL real type
   */
  template<typename TMAT,
           typename ALLOC1,
           typename ALLOC2,
           typename TREAL,
           typename = std::enable_if_t<qmc_allocator_traits<ALLOC1>::is_host_accessible>,
           typename = std::enable_if_t<qmc_allocator_traits<ALLOC2>::is_host_accessible>>
  inline std::enable_if_t<std::is_same<T_FP, TMAT>::value> invert_transpose(const Matrix<TMAT, ALLOC1>& amat,
                                                                            Matrix<TMAT, ALLOC2>& invMat,
                                                                            std::complex<TREAL>& LogDet)
  {
    const int n   = invMat.rows();
    const int lda = invMat.cols();
    simd::transpose(amat.data(), n, amat.cols(), invMat.data(), n, lda);
    computeInvertAndLog(invMat.data(), n, lda, LogDet);
  }

  /** compute the inverse of the transpose of matrix A and its determinant value in log
   * when T_FP and TMAT are not the same and need scratch space psiM_fp
   * @tparam TMAT matrix value type
   * @tparam TREAL real type
   */
  template<typename TMAT,
           typename ALLOC1,
           typename ALLOC2,
           typename TREAL,
           typename = std::enable_if_t<qmc_allocator_traits<ALLOC1>::is_host_accessible>,
           typename = std::enable_if_t<qmc_allocator_traits<ALLOC2>::is_host_accessible>>
  inline std::enable_if_t<!std::is_same<T_FP, TMAT>::value> invert_transpose(const Matrix<TMAT, ALLOC1>& amat,
                                                                             Matrix<TMAT, ALLOC2>& invMat,
                                                                             std::complex<TREAL>& LogDet)
  {
    const int n   = invMat.rows();
    const int lda = invMat.cols();
    psiM_fp.resize(n, lda);
    simd::transpose(amat.data(), n, amat.cols(), psiM_fp.data(), n, lda);
    computeInvertAndLog(psiM_fp.data(), n, lda, LogDet);
    invMat = psiM_fp;
  }
};
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_DIRAC_MATRIX_H
