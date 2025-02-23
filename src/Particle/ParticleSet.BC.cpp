//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/**@file ParticleSet.BC.cpp
 * @brief definition of functions controlling Boundary Conditions
 */
#include "Particle/ParticleSet.h"
#include "Particle/FastParticleOperators.h"
#include "Message/OpenMP.h"
#include "LongRange/StructFact.h"

namespace qmcplusplus
{
/** Creating StructureFactor
 *
 * Currently testing only 1 component for PBCs.
 */
void ParticleSet::createSK()
{
  if (SK)
    throw std::runtime_error("Report bug! SK has already been created. Unexpected call sequence.");

  if (Lattice.explicitly_defined)
    convert2Cart(R); //make sure that R is in Cartesian coordinates

  if (Lattice.SuperCellEnum != SUPERCELL_OPEN)
  {
    Lattice.SetLRCutoffs(Lattice.Rv);
    LRBox        = Lattice;
    bool changed = false;
    if (Lattice.SuperCellEnum == SUPERCELL_SLAB && Lattice.VacuumScale != 1.0)
    {
      LRBox.R(2, 0) *= Lattice.VacuumScale;
      LRBox.R(2, 1) *= Lattice.VacuumScale;
      LRBox.R(2, 2) *= Lattice.VacuumScale;
      changed = true;
    }
    else if (Lattice.SuperCellEnum == SUPERCELL_WIRE && Lattice.VacuumScale != 1.0)
    {
      LRBox.R(1, 0) *= Lattice.VacuumScale;
      LRBox.R(1, 1) *= Lattice.VacuumScale;
      LRBox.R(1, 2) *= Lattice.VacuumScale;
      LRBox.R(2, 0) *= Lattice.VacuumScale;
      LRBox.R(2, 1) *= Lattice.VacuumScale;
      LRBox.R(2, 2) *= Lattice.VacuumScale;
      changed = true;
    }
    LRBox.reset();
    LRBox.SetLRCutoffs(LRBox.Rv);
    LRBox.printCutoffs(app_log());

    if (changed)
    {
      app_summary() << "  Simulation box changed by vacuum supercell conditions" << std::endl;
      app_log() << "--------------------------------------- " << std::endl;
      LRBox.print(app_log());
      app_log() << "--------------------------------------- " << std::endl;
    }

    app_log() << "\n  Creating Structure Factor for periodic systems " << LRBox.LR_kc << std::endl;
    SK = std::make_unique<StructFact>(mySpecies.size(), TotalNum, LRBox, LRBox.LR_kc);
  }

  //set the mass array
  int beforemass = mySpecies.numAttributes();
  int massind    = mySpecies.addAttribute("mass");
  if (beforemass == massind)
  {
    app_log() << "  ParticleSet::createSK setting mass of  " << getName() << " to 1.0" << std::endl;
    for (int ig = 0; ig < mySpecies.getTotalNum(); ++ig)
      mySpecies(massind, ig) = 1.0;
  }
  for (int iat = 0; iat < GroupID.size(); iat++)
    Mass[iat] = mySpecies(massind, GroupID[iat]);

  coordinates_->setAllParticlePos(R);
}

void ParticleSet::turnOnPerParticleSK()
{
  if (SK)
    SK->turnOnStorePerParticle(*this);
  else
    APP_ABORT(
        "ParticleSet::turnOnPerParticleSK trying to turn on per particle storage in SK but SK has not been created.");
}

bool ParticleSet::getPerParticleSKState() const
{
  bool isPerParticleOn = false;
  if (SK)
    isPerParticleOn = SK->isStorePerParticle();
  return isPerParticleOn;
}

void ParticleSet::convert(const ParticlePos_t& pin, ParticlePos_t& pout)
{
  if (pin.getUnit() == pout.getUnit())
  {
    pout = pin;
    return;
  }
  if (pin.getUnit() == PosUnit::Lattice)
  //convert to CartesianUnit
  {
    ConvertPosUnit<ParticlePos_t, Tensor_t, DIM>::apply(pin, Lattice.R, pout, 0, pin.size());
  }
  else
  //convert to LatticeUnit
  {
    ConvertPosUnit<ParticlePos_t, Tensor_t, DIM>::apply(pin, Lattice.G, pout, 0, pin.size());
  }
}

void ParticleSet::convert2Unit(const ParticlePos_t& pin, ParticlePos_t& pout)
{
  pout.setUnit(PosUnit::Lattice);
  if (pin.getUnit() == PosUnit::Lattice)
    pout = pin;
  else
    ConvertPosUnit<ParticlePos_t, Tensor_t, DIM>::apply(pin, Lattice.G, pout, 0, pin.size());
}

void ParticleSet::convert2Cart(const ParticlePos_t& pin, ParticlePos_t& pout)
{
  pout.setUnit(PosUnit::Cartesian);
  if (pin.getUnit() == PosUnit::Cartesian)
    pout = pin;
  else
    ConvertPosUnit<ParticlePos_t, Tensor_t, DIM>::apply(pin, Lattice.R, pout, 0, pin.size());
}

void ParticleSet::convert2Unit(ParticlePos_t& pinout)
{
  if (pinout.getUnit() == PosUnit::Lattice)
    return;
  else
  {
    pinout.setUnit(PosUnit::Lattice);
    ConvertPosUnit<ParticlePos_t, Tensor_t, DIM>::apply(pinout, Lattice.G, 0, pinout.size());
  }
}

void ParticleSet::convert2Cart(ParticlePos_t& pinout)
{
  if (pinout.getUnit() == PosUnit::Cartesian)
    return;
  else
  {
    pinout.setUnit(PosUnit::Cartesian);
    ConvertPosUnit<ParticlePos_t, Tensor_t, DIM>::apply(pinout, Lattice.R, 0, pinout.size());
  }
}

void ParticleSet::applyBC(const ParticlePos_t& pin, ParticlePos_t& pout) { applyBC(pin, pout, 0, pin.size()); }

void ParticleSet::applyBC(const ParticlePos_t& pin, ParticlePos_t& pout, int first, int last)
{
  if (pin.getUnit() == PosUnit::Cartesian)
  {
    if (pout.getUnit() == PosUnit::Cartesian)
      ApplyBConds<ParticlePos_t, Tensor_t, DIM>::Cart2Cart(pin, Lattice.G, Lattice.R, pout, first, last);
    else if (pout.getUnit() == PosUnit::Lattice)
      ApplyBConds<ParticlePos_t, Tensor_t, DIM>::Cart2Unit(pin, Lattice.G, pout, first, last);
    else
      throw std::runtime_error("Unknown unit conversion");
  }
  else if (pin.getUnit() == PosUnit::Lattice)
  {
    if (pout.getUnit() == PosUnit::Cartesian)
      ApplyBConds<ParticlePos_t, Tensor_t, DIM>::Unit2Cart(pin, Lattice.R, pout, first, last);
    else if (pout.getUnit() == PosUnit::Lattice)
      ApplyBConds<ParticlePos_t, Tensor_t, DIM>::Unit2Unit(pin, pout, first, last);
    else
      throw std::runtime_error("Unknown unit conversion");
  }
  else
    throw std::runtime_error("Unknown unit conversion");
}

void ParticleSet::applyBC(ParticlePos_t& pos)
{
  if (pos.getUnit() == PosUnit::Lattice)
  {
    ApplyBConds<ParticlePos_t, Tensor_t, DIM>::Unit2Unit(pos, 0, TotalNum);
  }
  else
  {
    ApplyBConds<ParticlePos_t, Tensor_t, DIM>::Cart2Cart(pos, Lattice.G, Lattice.R, 0, TotalNum);
  }
}

void ParticleSet::applyMinimumImage(ParticlePos_t& pinout)
{
  if (Lattice.SuperCellEnum == SUPERCELL_OPEN)
    return;
  for (int i = 0; i < pinout.size(); ++i)
    Lattice.applyMinimumImage(pinout[i]);
}

void ParticleSet::convert2UnitInBox(const ParticlePos_t& pin, ParticlePos_t& pout)
{
  pout.setUnit(PosUnit::Lattice);
  convert2Unit(pin, pout); // convert to crystalline unit
  put2box(pout);
}

void ParticleSet::convert2CartInBox(const ParticlePos_t& pin, ParticlePos_t& pout)
{
  convert2UnitInBox(pin, pout); // convert to crystalline unit
  convert2Cart(pout);
}
} // namespace qmcplusplus
