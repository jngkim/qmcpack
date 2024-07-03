//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corporation
//                    Alex Wells, alex.wells@intel.com, Intel Corporation
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corporation
//////////////////////////////////////////////////////////////////////////////////////

#include "SplineC2RHelper.hpp"
#include "ComplexView.hpp"
#include "Utilities/Addressing.hpp"
#include "QMCWaveFunctions/BsplineFactory/contraction_helper.hpp"

namespace qmcplusplus
{
namespace C2R
{
/** assign_vgl
   */
template<typename ST, typename TT>
void assign_vgl_simd_load2(ST x, ST y, ST z,
                           SPOSet::ValueVector& psi,
                           SPOSet::GradVector& dpsi,
                           SPOSet::ValueVector& d2psi,
                           const TT* restrict myV,
                           const TT* restrict myG,
                           const TT* restrict myH,
                           int spline_padded_size,
                           const Tensor<ST, 3>& G,
                           const Tensor<ST, 3>& GGt,
                           const ST* restrict myKcart_ptr,
                           const ST* restrict mKK,
                           int myKcart_size,
                           int myKcart_padded_size,
                           int first,
                           int last,
                           int nComplexBands)
{
  // protect last
  last = last > myKcart_size ? myKcart_size : last;

  const ST g00 = G(0), g01 = G(1), g02 = G(2), g10 = G(3),
           g11 = G(4), g12 = G(5), g20 = G(6), g21 = G(7),
           g22 = G(8);

  const ST symGG[6] = {GGt[0], GGt[1] + GGt[3], GGt[2] + GGt[6], GGt[4], GGt[5] + GGt[7], GGt[8]};

  const ST* restrict k0 = myKcart_ptr;
  ASSUME_ALIGNED(k0);
  const ST* restrict k1 = myKcart_ptr + myKcart_padded_size;
  ASSUME_ALIGNED(k1);
  const ST* restrict k2 = myKcart_ptr + myKcart_padded_size * 2;
  ASSUME_ALIGNED(k2);

  // Rather than accessing an multiple elements of an array of ST with 
  // non-unit stride indices [2*j], [2*j+1], view the data as an array of
  // ComplexNumber<ST> with .real() and .imag() data members 
  // using linear index [j].  This will allow a specialized ComplexView<float>
  // to do unit stride 64bit loads|stores to quickly access 2 32bit floats
  // which otherwise would have been gathers|scatters.
  // It also simplifies the indexing inside the loop
  ComplexView<const TT> g0(myG);
  ComplexView<const TT> g1(myG + spline_padded_size    );
  ComplexView<const TT> g2(myG + spline_padded_size * 2);
  ComplexView<const TT> h00(myH);
  ComplexView<const TT> h01(myH + spline_padded_size    );
  ComplexView<const TT> h02(myH + spline_padded_size * 2);
  ComplexView<const TT> h11(myH + spline_padded_size * 3);
  ComplexView<const TT> h12(myH + spline_padded_size * 4);
  ComplexView<const TT> h22(myH + spline_padded_size * 5);
  
  ComplexView<const TT> myV_cn(myV);

  const int first_spo = 0;

  using ValueType = SPOSet::ValueType;
  using GradType = TinyVector<SPOSet::ValueType,3>;

  ComplexView<ValueType> psi_cn(psi.data()+first_spo);
  ComplexView<ValueType> d2psi_cn(d2psi.data()+first_spo);
  // Prefer using view with structured type vs. multiple individual array
  // accesses to allow the compiler to share a single base address + index +
  // multiple compile time known offsets to members in the type.
  ComplexView<GradType> dpsi_cn(dpsi.data() + first_spo);

  // NOTE: don't use size_t as the index it can force address calculations
  // to use 64bit offsets
  int jend = std::min(nComplexBands, last);
#pragma omp simd
  for (int j = first; j < jend; j++)
  {
    const ST kX    = k0[j];
    const ST kY    = k1[j];
    const ST kZ    = k2[j];
    auto val = myV_cn[j];

    //phase
    ST s, c;
    qmcplusplus::sincos(-(x * kX + y * kY + z * kZ), &s, &c);

    //dot(PrimLattice.G,myG[j])
    auto cg0 = g0[j];
    auto cg1 = g1[j];
    auto cg2 = g2[j];
    const ST dX_r = g00 * cg0.real() + g01 * cg1.real() + g02 * cg2.real();
    const ST dY_r = g10 * cg0.real() + g11 * cg1.real() + g12 * cg2.real();
    const ST dZ_r = g20 * cg0.real() + g21 * cg1.real() + g22 * cg2.real();

    const ST dX_i = g00 * cg0.imag() + g01 * cg1.imag() + g02 * cg2.imag();
    const ST dY_i = g10 * cg0.imag() + g11 * cg1.imag() + g12 * cg2.imag();
    const ST dZ_i = g20 * cg0.imag() + g21 * cg1.imag() + g22 * cg2.imag();

    // \f$\nabla \psi_r + {\bf k}\psi_i\f$
    const ST gX_r = dX_r + val.imag() * kX;
    const ST gY_r = dY_r + val.imag() * kY;
    const ST gZ_r = dZ_r + val.imag() * kZ;
    const ST gX_i = dX_i - val.real() * kX;
    const ST gY_i = dY_i - val.real() * kY;
    const ST gZ_i = dZ_i - val.real() * kZ;

    auto ch00 = h00[j];
    auto ch01 = h01[j];
    auto ch02 = h02[j];
    auto ch11 = h11[j];
    auto ch12 = h12[j];
    auto ch22 = h22[j];
    const ST lcart_r = SymTrace(ch00.real(), ch01.real(), ch02.real(), ch11.real(), ch12.real(), ch22.real(), symGG);
    const ST lcart_i = SymTrace(ch00.imag(), ch01.imag(), ch02.imag(), ch11.imag(), ch12.imag(), ch22.imag(), symGG);
    // Avoid multiplying by 2.0, which would require loading a constant from 
    // memory and broadcasting it into a SIMD register, by simplying adding or
    // subtracting the term, which is already in the register, twice.
    // const ST lap_r   = lcart_r + mKK[j] * val.real() + ST(2) * (kX * dX_i + kY * dY_i + kZ * dZ_i);
    // const ST lap_i   = lcart_i + mKK[j] * val.imag() - ST(2) * (kX * dX_r + kY * dY_r + kZ * dZ_r);
    const ST lap_r_term = (kX * dX_i + kY * dY_i + kZ * dZ_i);
    const ST lap_r   = lcart_r + mKK[j] * val.real() + lap_r_term + lap_r_term;
    const ST lap_i_term = (kX * dX_r + kY * dY_r + kZ * dZ_r);
    const ST lap_i   = lcart_i + mKK[j] * val.imag() - lap_i_term - lap_i_term;

    psi_cn[j] = ComplexNumber<ValueType>{static_cast<ValueType>(c * val.real() - s * val.imag()), static_cast<ValueType>(c * val.imag() + s * val.real())};
    d2psi_cn[j] = ComplexNumber<ValueType>{static_cast<ValueType>(c * lap_r - s * lap_i), static_cast<ValueType>(c * lap_i + s * lap_r)};

    // Avoid 64bit scatters by manipulating the index to ensure a 32bit offset
    int j32 = ensure_32bit_offset<ComplexNumber<GradType>>(j);
    // Could dpsi be changed to SOA, this generates 6 scatters
    auto & cn_grad = dpsi_cn[j32];
    auto & grad_re = cn_grad.real();
    grad_re[0] = c * gX_r - s * gX_i;
    grad_re[1] = c * gY_r - s * gY_i;
    grad_re[2] = c * gZ_r - s * gZ_i;
    auto & grad_im = cn_grad.imag();
    grad_im[0] = c * gX_i + s * gX_r;
    grad_im[1] = c * gY_i + s * gY_r;
    grad_im[2] = c * gZ_i + s * gZ_r;
  }

  const int offset_re = static_cast<int>(first_spo + nComplexBands);
  // Simplify indexing by baking offset to the start of the non complex numbers
  // into a pointer vs. allowing it to possibly be incorporated in the final
  // address calculation causing 64bit indices.  64bit indices for indirect 
  // accesses can cause 2 gather|scatter instructions vs. 1 with 32bit indices.
  auto * dpsi_re = rebasePointer(dpsi.data() + offset_re);
  auto * d2psi_re = rebasePointer(d2psi.data() + offset_re);
  auto *psi_re = rebasePointer(psi.data() + offset_re);

#pragma omp simd
  for (int j = std::max(nComplexBands, first); j < last; j++)
  {
    const ST kX    = k0[j];
    const ST kY    = k1[j];
    const ST kZ    = k2[j];
    auto val = myV_cn[j];

    //phase
    ST s, c;
    qmcplusplus::sincos(-(x * kX + y * kY + z * kZ), &s, &c);

    //dot(PrimLattice.G,myG[j])
    auto cg0 = g0[j];
    auto cg1 = g1[j];
    auto cg2 = g2[j];
    const ST dX_r = g00 * cg0.real() + g01 * cg1.real() + g02 * cg2.real();
    const ST dY_r = g10 * cg0.real() + g11 * cg1.real() + g12 * cg2.real();
    const ST dZ_r = g20 * cg0.real() + g21 * cg1.real() + g22 * cg2.real();

    const ST dX_i = g00 * cg0.imag() + g01 * cg1.imag() + g02 * cg2.imag();
    const ST dY_i = g10 * cg0.imag() + g11 * cg1.imag() + g12 * cg2.imag();
    const ST dZ_i = g20 * cg0.imag() + g21 * cg1.imag() + g22 * cg2.imag();

    // \f$\nabla \psi_r + {\bf k}\psi_i\f$
    const ST gX_r = dX_r + val.imag() * kX;
    const ST gY_r = dY_r + val.imag() * kY;
    const ST gZ_r = dZ_r + val.imag() * kZ;
    const ST gX_i = dX_i - val.real() * kX;
    const ST gY_i = dY_i - val.real() * kY;
    const ST gZ_i = dZ_i - val.real() * kZ;

    const size_t psiIndex = first_spo + nComplexBands + j;
    psi_re[j]         = c * val.real() - s * val.imag();

    // Avoid 64bit scatters by manipulating the index to ensure a 32bit offset
    int j32 = ensure_32bit_offset<GradType>(j);
    auto &grad_re = dpsi_re[j32];
    grad_re[0]     = c * gX_r - s * gX_i;
    grad_re[1]     = c * gY_r - s * gY_i;
    grad_re[2]     = c * gZ_r - s * gZ_i;

    auto ch00 = h00[j];
    auto ch01 = h01[j];
    auto ch02 = h02[j];
    auto ch11 = h11[j];
    auto ch12 = h12[j];
    auto ch22 = h22[j];

    const ST lcart_r = SymTrace(ch00.real(), ch01.real(), ch02.real(), ch11.real(), ch12.real(), ch22.real(), symGG);
    const ST lcart_i = SymTrace(ch00.imag(), ch01.imag(), ch02.imag(), ch11.imag(), ch12.imag(), ch22.imag(), symGG);
    // Avoid multiplying by 2.0, which would require loading a constant from 
    // memory and broadcasting it into a SIMD register, by simplying adding the
    // term, which is already in the register, to itself.
    // const ST lap_r   = lcart_r + mKK[j] * val.real() + ST(2) * (kX * dX_i + kY * dY_i + kZ * dZ_i);
    // const ST lap_i   = lcart_i + mKK[j] * val.imag() - ST(2) * (kX * dX_r + kY * dY_r + kZ * dZ_r);
    const ST lap_r_term   = (kX * dX_i + kY * dY_i + kZ * dZ_i);
    const ST lap_r   = lcart_r + mKK[j] * val.real() + lap_r_term + lap_r_term;
    const ST lap_i_term   = (kX * dX_r + kY * dY_r + kZ * dZ_r);
    const ST lap_i   = lcart_i + mKK[j] * val.imag() - lap_i_term - lap_i_term;

    d2psi_re[j]  = c * lap_r - s * lap_i;
  }
}

template<typename ST, typename TT>
void assign_vgl_simd_base(ST x, ST y, ST z,
                          SPOSet::ValueVector& psi,
                          SPOSet::GradVector& dpsi,
                          SPOSet::ValueVector& d2psi,
                          const TT* restrict myV,
                          const TT* restrict myG,
                          const TT* restrict myH,
                          int spline_padded_size,
                          const Tensor<ST, 3>& G,
                          const Tensor<ST, 3>& GGt,
                          const ST* restrict myKcart_ptr,
                          const ST* restrict mKK,
                          int myKcart_size,
                          int myKcart_padded_size,
                          int first,
                          int last,
                          int nComplexBands)
{
  // protect last
  last = last > myKcart_size ? myKcart_size : last;

  constexpr ST two(2);
  const ST g00 = G(0), g01 = G(1), g02 = G(2), g10 = G(3),
           g11 = G(4), g12 = G(5), g20 = G(6), g21 = G(7),
           g22 = G(8);

  const ST symGG[6] = {GGt[0], GGt[1] + GGt[3], GGt[2] + GGt[6], GGt[4], GGt[5] + GGt[7], GGt[8]};

  const ST* restrict k0 = myKcart_ptr;
  ASSUME_ALIGNED(k0);
  const ST* restrict k1 = myKcart_ptr + myKcart_padded_size;
  ASSUME_ALIGNED(k1);
  const ST* restrict k2 = myKcart_ptr + myKcart_padded_size * 2;
  ASSUME_ALIGNED(k2);

  const TT* restrict g0(myG);
  ASSUME_ALIGNED(g0);
  const TT* restrict g1(myG + spline_padded_size    );
  ASSUME_ALIGNED(g1);
  const TT* restrict g2(myG + spline_padded_size * 2);
  ASSUME_ALIGNED(g2);
  const TT* restrict h00(myH);
  ASSUME_ALIGNED(h00);
  const TT* restrict h01(myH + spline_padded_size    );
  ASSUME_ALIGNED(h01);
  const TT* restrict h02(myH + spline_padded_size * 2);
  ASSUME_ALIGNED(h02);
  const TT* restrict h11(myH + spline_padded_size * 3);
  ASSUME_ALIGNED(h11);
  const TT* restrict h12(myH + spline_padded_size * 4);
  ASSUME_ALIGNED(h12);
  const TT* restrict h22(myH + spline_padded_size * 5);
  ASSUME_ALIGNED(h22);

  const int first_spo = 0;

#pragma omp simd
  for (size_t j = first; j < std::min(nComplexBands, last); j++)
  {
    const size_t jr = j << 1;
    const size_t ji = jr + 1;

    const ST kX    = k0[j];
    const ST kY    = k1[j];
    const ST kZ    = k2[j];
    const ST val_r = myV[jr];
    const ST val_i = myV[ji];

    //phase
    ST s, c;
    qmcplusplus::sincos(-(x * kX + y * kY + z * kZ), &s, &c);

    //dot(PrimLattice.G,myG[j])
    const ST dX_r = g00 * g0[jr] + g01 * g1[jr] + g02 * g2[jr];
    const ST dY_r = g10 * g0[jr] + g11 * g1[jr] + g12 * g2[jr];
    const ST dZ_r = g20 * g0[jr] + g21 * g1[jr] + g22 * g2[jr];

    const ST dX_i = g00 * g0[ji] + g01 * g1[ji] + g02 * g2[ji];
    const ST dY_i = g10 * g0[ji] + g11 * g1[ji] + g12 * g2[ji];
    const ST dZ_i = g20 * g0[ji] + g21 * g1[ji] + g22 * g2[ji];

    // \f$\nabla \psi_r + {\bf k}\psi_i\f$
    const ST gX_r = dX_r + val_i * kX;
    const ST gY_r = dY_r + val_i * kY;
    const ST gZ_r = dZ_r + val_i * kZ;
    const ST gX_i = dX_i - val_r * kX;
    const ST gY_i = dY_i - val_r * kY;
    const ST gZ_i = dZ_i - val_r * kZ;

    const ST lcart_r = SymTrace(h00[jr], h01[jr], h02[jr], h11[jr], h12[jr], h22[jr], symGG);
    const ST lcart_i = SymTrace(h00[ji], h01[ji], h02[ji], h11[ji], h12[ji], h22[ji], symGG);
    const ST lap_r   = lcart_r + mKK[j] * val_r + two * (kX * dX_i + kY * dY_i + kZ * dZ_i);
    const ST lap_i   = lcart_i + mKK[j] * val_i - two * (kX * dX_r + kY * dY_r + kZ * dZ_r);

    const size_t psiIndex = first_spo + jr;
    psi[psiIndex]         = c * val_r - s * val_i;
    psi[psiIndex + 1]     = c * val_i + s * val_r;
    d2psi[psiIndex]       = c * lap_r - s * lap_i;
    d2psi[psiIndex + 1]   = c * lap_i + s * lap_r;
    dpsi[psiIndex][0]     = c * gX_r - s * gX_i;
    dpsi[psiIndex][1]     = c * gY_r - s * gY_i;
    dpsi[psiIndex][2]     = c * gZ_r - s * gZ_i;
    dpsi[psiIndex + 1][0] = c * gX_i + s * gX_r;
    dpsi[psiIndex + 1][1] = c * gY_i + s * gY_r;
    dpsi[psiIndex + 1][2] = c * gZ_i + s * gZ_r;
  }

#pragma omp simd
  for (size_t j = std::max(nComplexBands, first); j < last; j++)
  {
    const size_t jr = j << 1;
    const size_t ji = jr + 1;

    const ST kX    = k0[j];
    const ST kY    = k1[j];
    const ST kZ    = k2[j];
    const ST val_r = myV[jr];
    const ST val_i = myV[ji];

    //phase
    ST s, c;
    qmcplusplus::sincos(-(x * kX + y * kY + z * kZ), &s, &c);

    //dot(PrimLattice.G,myG[j])
    const ST dX_r = g00 * g0[jr] + g01 * g1[jr] + g02 * g2[jr];
    const ST dY_r = g10 * g0[jr] + g11 * g1[jr] + g12 * g2[jr];
    const ST dZ_r = g20 * g0[jr] + g21 * g1[jr] + g22 * g2[jr];

    const ST dX_i = g00 * g0[ji] + g01 * g1[ji] + g02 * g2[ji];
    const ST dY_i = g10 * g0[ji] + g11 * g1[ji] + g12 * g2[ji];
    const ST dZ_i = g20 * g0[ji] + g21 * g1[ji] + g22 * g2[ji];

    // \f$\nabla \psi_r + {\bf k}\psi_i\f$
    const ST gX_r = dX_r + val_i * kX;
    const ST gY_r = dY_r + val_i * kY;
    const ST gZ_r = dZ_r + val_i * kZ;
    const ST gX_i = dX_i - val_r * kX;
    const ST gY_i = dY_i - val_r * kY;
    const ST gZ_i = dZ_i - val_r * kZ;

    const size_t psiIndex = first_spo + nComplexBands + j;
    psi[psiIndex]         = c * val_r - s * val_i;
    dpsi[psiIndex][0]     = c * gX_r - s * gX_i;
    dpsi[psiIndex][1]     = c * gY_r - s * gY_i;
    dpsi[psiIndex][2]     = c * gZ_r - s * gZ_i;

    const ST lcart_r = SymTrace(h00[jr], h01[jr], h02[jr], h11[jr], h12[jr], h22[jr], symGG);
    const ST lcart_i = SymTrace(h00[ji], h01[ji], h02[ji], h11[ji], h12[ji], h22[ji], symGG);
    const ST lap_r   = lcart_r + mKK[j] * val_r + two * (kX * dX_i + kY * dY_i + kZ * dZ_i);
    const ST lap_i   = lcart_i + mKK[j] * val_i - two * (kX * dX_r + kY * dY_r + kZ * dZ_r);
    d2psi[psiIndex]  = c * lap_r - s * lap_i;
  }
}

void assign_vgl_simd(float x, float y, float z,
                              SPOSet::ValueVector& psi,
                              SPOSet::GradVector& dpsi,
                              SPOSet::ValueVector& d2psi,
                              const float* myV,
                              const float* myG,
                              const float* myH,
                              int spline_padded_size,
                              const Tensor<float, 3>& G,
                              const Tensor<float, 3>& GGt,
                              const float* myKcart_ptr,
                              const float* mKK,
                              int myKcart_size,
                              int myKcart_padded_size,
                              int first,
                              int last,
                              int nComplexBands)
{
  assign_vgl_simd_load2(x,y,z,psi,dpsi,d2psi,myV,myG,myH,  spline_padded_size, G, GGt,
                       myKcart_ptr, mKK, myKcart_size, myKcart_padded_size, first, last, nComplexBands);
}


void assign_vgl_simd(double x, 
                     double y, 
                     double z,
                     SPOSet::ValueVector& psi,
                     SPOSet::GradVector& dpsi,
                     SPOSet::ValueVector& d2psi,
                     const double* myV,
                     const double* myG,
                     const double* myH,
                     int spline_padded_size,
                     const Tensor<double, 3>& G,
                     const Tensor<double, 3>& GGt,
                     const double* myKcart_ptr,
                     const double* mKK,
                     int myKcart_size,
                     int myKcart_padded_size,
                     int first,
                     int last,
                     int nComplexBands)
{

  assign_vgl_simd_load2(x,y,z,psi,dpsi,d2psi,myV,myG,myH,  spline_padded_size, G, GGt,
                       myKcart_ptr, mKK, myKcart_size, myKcart_padded_size, first, last, nComplexBands);
}


}
}
