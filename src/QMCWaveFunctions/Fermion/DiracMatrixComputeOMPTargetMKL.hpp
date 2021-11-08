//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_DIRAC_MATRIX_COMPUTE_OMPTARGET_MKL_H
#define QMCPLUSPLUS_DIRAC_MATRIX_COMPUTE_OMPTARGET_MKL_H

#include "mkl.h"
#include "mkl_omp_offload.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "OMPTarget/OMPallocator.hpp"
#include "Platforms/PinnedAllocator.h"
#include "DiracMatrix.h"
#include "type_traits/complex_help.hpp"
#include "type_traits/template_types.hpp"
#include "Message/OpenMP.h"
#include "CPU/SIMD/simd.hpp"
#include "ResourceCollection.h"

namespace qmcplusplus
{
namespace mkl_ompt
{
//POC of MKL OpenMP API
//#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetrf)) match(construct={target variant dispatch}, device={arch(gen)})
//void dgetrf(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info) NOTHROW;
//
//#pragma omp declare variant (MKL_LAPACK_OPENMP_OFFLOAD(dgetri)) match(construct={target variant dispatch}, device={arch(gen)})
//void dgetri(const MKL_INT* n, double* a, const MKL_INT* lda, const MKL_INT* ipiv, double* work, const MKL_INT* lwork,
//            MKL_INT* info) NOTHROW;
//

  template<typename T>
  inline void getrf(lapack_int m, lapack_int n, T* A, lapack_int lda, lapack_int* ipiv, lapack_int* info)
  {
    const int dnum=0;
#pragma omp target variant dispatch device(dnum) use_device_ptr(A, ipiv, info)
    {
      dgetrf(&m, &n, A, &lda, ipiv, info);
    }
  }

  template<typename T>
  inline void getri(lapack_int m, T* A, lapack_int lda, lapack_int* ipiv, T* work, lapack_int lwork, lapack_int* info)
  {
    const int dnum=0;
#pragma omp target variant dispatch device(dnum) use_device_ptr(A, ipiv, work, info)
    {
      dgetri(&m, A, &lda, ipiv, work, &lwork, info);
    }
  }

  template<typename T>
    inline void getrf_batch(lapack_int m, lapack_int n, T* A, lapack_int lda, lapack_int stride_a, 
        lapack_int* ipiv, lapack_int stride_ipiv, lapack_int batchsize, lapack_int* info)
    {
      const int dnum=0;
#pragma omp target variant dispatch device(dnum) use_device_ptr(A, ipiv, info)
      {
        dgetrf_batch_strided(&m,&n,A,&lda,&stride_a,ipiv,&stride_ipiv, &batchsize, info);
      }
    }

  template<typename T>
    inline void getri_batch(lapack_int m, 
        T* A, lapack_int lda, lapack_int stride_a, 
        lapack_int* ipiv, lapack_int stride_ipiv, 
        T* invA, lapack_int invlda, lapack_int stride_inv, 
        lapack_int batchsize, lapack_int* info)
    {
      const int dnum=0;
#pragma omp target variant dispatch device(dnum) use_device_ptr(A, invA, ipiv, info)
      {
        dgetri_oop_batch_strided(&m,A,&lda,&stride_a,ipiv,&stride_ipiv,invA,&invlda,&stride_inv, &batchsize, info);
      }
    }

  template<typename T1, typename T2>
   inline void transposeTo(const T1* restrict in, int n, int lda,
       T2* restrict out, int m, int ldb)
    {
#pragma omp target teams distribute parallel for collapse(2) 
      for(int i=0; i<n; ++i)
        for(int j=0; j<m; ++j)
          out[i*ldb+j]=in[i+j*lda];
    }

  template<typename T1, typename T2>
   inline void transposeFrom(const T1* restrict in, int n, int lda,
       T2* restrict out, int m, int ldb)
    {
#pragma omp target teams distribute parallel for collapse(2) map(from:out[:m*ldb])
      for(int i=0; i<n; ++i)
        for(int j=0; j<m; ++j)
          out[i*ldb+j]=in[i+j*lda];
    }
}

template<typename T_FP>
inline std::complex<T_FP> computeLogDet2(const T_FP* restrict inv_mat, int n, int lda, 
    const lapack_int* restrict pivot)
{
  std::complex<T_FP> logdet{};
#pragma omp target teams distribute parallel for reduction(+:logdet) 
  for (size_t i = 0; i < n; i++)
    logdet += std::log(std::complex<T_FP>((pivot[i] == i + 1) ? inv_mat[i*lda+i] : -inv_mat[i*lda+i]));
  return logdet;
}

template<typename T_FP>
inline void computeLogDetBatch(const T_FP* restrict inv_mat, int n, int lda, 
    const lapack_int* restrict pivot, std::complex<T_FP>* log_values, int batchsize)
{

  if(n<32)
  {
#pragma omp target teams distribute parallel for map(from:log_values[:batchsize])
    for(int iw=0; iw<batchsize; ++iw)
    {
      const T_FP* restrict a=inv_mat+iw*n*lda;
      std::complex<T_FP> logdet{};
#pragma omp simd reduction(+:logdet) 
      for (size_t i = 0; i < n; i++)
        logdet += std::log(std::complex<T_FP>((pivot[lda+i] == i + 1) ? a[i*lda+i] : -a[i*lda+i]));
      log_values[iw]=logdet;
    }
  }
  else
  {
#pragma omp target teams distribute map(from:log_values[:batchsize])
    for(int iw=0; iw<batchsize; ++iw)
    {
      const T_FP* restrict a=inv_mat+iw*n*lda;
      std::complex<T_FP> logdet{};
#pragma omp parallel for reduction(+:logdet) 
      for (size_t i = 0; i < n; i++)
        logdet += std::log(std::complex<T_FP>((pivot[lda+i] == i + 1) ? a[i*lda+i] : -a[i*lda+i]));
      log_values[iw]=logdet;
    }
  }
}

/** class to compute matrix inversion and the log value of determinant
 *  of a batch of DiracMatrixes.
 *
 *  @tparam VALUE_FP the datatype used in the actual computation of the matrix
 *  
 *  There is one per crowd not one per MatrixUpdateEngine.
 *  this puts ownership of the scratch resources in a sensible place.
 *  
 *  Currently this is CPU only but its external API is somewhat written to
 *  enforce the passing Dual data objects as arguments.  Except for the single
 *  particle API log_value which is not Dual type but had better have an address in a OMPtarget
 *  mapped region if target is used with it. This makes this API incompatible to
 *  that used by MatrixDelayedUpdateCuda and DiracMatrixComputeCUDA.
 */
template<typename VALUE_FP>
class DiracMatrixComputeOMPTarget : public Resource
{
public:
  using FullPrecReal = RealAlias<VALUE_FP>;
  using LogValue     = std::complex<FullPrecReal>;

  // This class only works with OMPallocator so explicitly call OffloadAllocator what it
  // is and not DUAL
  template<typename T>
  using OffloadPinnedAllocator = OMPallocator<T, PinnedAlignedAllocator<T>>;
  template<typename T>
  using OffloadPinnedMatrix = Matrix<T, OffloadPinnedAllocator<T>>;
  template<typename T>
  using OffloadPinnedVector = Vector<T, OffloadPinnedAllocator<T>>;

  // maybe you'll want a resource someday, then change here.
  using HandleResource = DummyResource;

private:
  int lwork_=0;
  //Unlike DiracMatrix.h these are contiguous packed representations of the Matrices
  // Contiguous Matrices for each walker, n^2 * nw  elements
  OffloadPinnedVector<VALUE_FP> psiM_fp_;
  OffloadPinnedVector<VALUE_FP> m_work_;  
  OffloadPinnedVector<VALUE_FP> LU_diags_fp_;
  OffloadPinnedVector<lapack_int> pivots_;
  OffloadPinnedVector<lapack_int> infos_;

  /** reset internal work space.
   *  My understanding might be off.
   *
   *  it smells that this is so complex.
   */
  inline void reset(OffloadPinnedVector<VALUE_FP>& psi_Ms, const int n, const int lda, const int batch_size)
  {
    const int nw = batch_size;
    LU_diags_fp_.resize(lda*batch_size);
    pivots_.resize(lda * nw);
    infos_.resize(nw);
    lwork_=n*lda;
    m_work_.resize(nw*lwork_);
  }

  /** reset internal work space for single walker case
   */
  inline void reset(OffloadPinnedMatrix<VALUE_FP>& psi_M, const int n, const int lda)
  {
    LU_diags_fp_.resize(lda);
    pivots_.resize(lda);
    infos_.resize(4);
    lwork_=n*lda;
    m_work_.resize(lwork_);
  }

  /** compute the inverse of invMat (in place) and the log value of determinant
   * \tparam TMAT value type of matrix
   * \param[inout] a_mat      the matrix
   * \param[in]    n          actual dimension of square matrix (no guarantee it really has full column rank)
   * \param[in]    lda        leading dimension of Matrix container
   * \param[out]   log_value  log a_mat before inversion
   */
  template<typename TMAT>
  inline void computeInvertAndLog(OffloadPinnedMatrix<TMAT>& a_mat, const int n, const int lda, LogValue& log_value)
  {

    if (lwork_ < lda)
      reset(a_mat, n, lda);

    mkl_ompt::getrf(n, n, a_mat.data(), lda, pivots_.data(),infos_.data());
    log_value=computeLogDet2(a_mat.data(), n, lda, pivots_.data());
    mkl_ompt::getri(n, a_mat.data(), lda, pivots_.data(), m_work_.data(), lwork_,infos_.data());

  }

  template<typename TMAT>
  inline void computeInvertAndLog(OffloadPinnedVector<TMAT>& psi_Ms,
                                  const int n,
                                  const int lda,
                                  OffloadPinnedVector<LogValue>& log_values)
  {
    std::cout << "ZZZ computeInvertAndLog(psi_Ms,n,lda,log_values) " << std::endl;
    const int nw = log_values.size();

    if(pivots_.size()<n*nw)
      reset(psi_Ms, n, lda, nw);

    mkl_ompt::getrf_batch(n,n,m_work_.data(),lda,lda*n,pivots_.data(),lda, nw, infos_.data());
    computeLogDetBatch(m_work_.data(),n,lda,pivots_.data(),log_values.data(),nw);
    //computeLogDetBatch(psi_Ms.data(),n,lda,pivots_.data(),lda,log_values);

    for (int iw = 0; iw < nw; ++iw)
    {
      VALUE_FP* LU_M = m_work_.data() + iw * n * n;
      log_values[iw]=computeLogDet2(LU_M, n, lda, pivots_.data()+lda*iw);
    }

    mkl_ompt::getri_batch(n, m_work_.data(), lda, n*lda, pivots_.data(), lda, 
        psi_Ms.data(), lda, lda*n, nw, infos_.data());

  }

  /// matrix inversion engine
  //DiracMatrix<VALUE_FP> detEng_;

public:
  DiracMatrixComputeOMPTarget() : Resource("DiracMatrixComputeOMPTarget") {}

  Resource* makeClone() const override { return new DiracMatrixComputeOMPTarget(*this); }

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
    const int n   = a_mat.rows();
    const int lda = a_mat.cols();
    const int ldb = inv_a_mat.cols();

    //FIXME 
    //mkl_ompt::transposeTo(a_mat.data(),n,lda,inv_a_mat.data(),n,ldb);
    simd::transpose(a_mat.data(), n, lda, inv_a_mat.data(), n, ldb);
    inv_a_mat.updateTo();
    computeInvertAndLog(inv_a_mat, n, ldb, log_value);
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

    psiM_fp_.resize(n * lda);

    simd::transpose(a_mat.data(), n, lda, psiM_fp_.data(), n, ldb);
    OffloadPinnedMatrix<VALUE_FP> psiM_fp_view(psiM_fp_, psiM_fp_.data(), n, lda);
    psiM_fp_view.updateTo();
    computeInvertAndLog(psiM_fp_view, n, lda, log_value);
    psiM_fp_view.updateFrom();
    simd::remapCopy(n, n, psiM_fp_.data(), lda, inv_a_mat.data(), ldb);

    //mkl_ompt::transposeFrom(psiM_fp_.data(),n,lda,inv_a_mat.data(),n,lda);
    //inv_a_mat.updateFrom();
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
    for (int iw = 0; iw < a_mats.size(); iw++)
    {
      auto& Ainv = inv_a_mats[iw].get();
      invert_transpose(resource,a_mats[iw].get(),Ainv,log_values[iw]);
    }
    
#if 0
    const int nw  = a_mats.size();
    const int n   = a_mats[0].get().rows();
    const int lda = a_mats[0].get().cols();
    const int ldb = inv_a_mats[0].get().cols();
    const int nsqr{n * lda};

    psiM_fp_.resize(n * lda * nw);
    m_work_.resize(n * lda * nw);

    for (int iw = 0; iw < nw; ++iw)
      simd::transpose(a_mats[iw].get().data(), n, lda, m_work_.data() + nsqr * iw, n, lda);

    m_work_.updateTo();
    computeInvertAndLog(psiM_fp_, n, lda, log_values);
    psiM_fp_.updateFrom();

    for (int iw = 0; iw < nw; ++iw)
    {
      simd::remapCopy(n, n, psiM_fp_.data() + nsqr * iw, lda, inv_a_mats[iw].get().data(), ldb);
    }
#endif
  }
};
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_DIRAC_MATRIX_COMPUTE_OMPTARGET_H
