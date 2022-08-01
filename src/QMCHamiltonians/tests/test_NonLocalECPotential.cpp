//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#include "catch.hpp"

#include "Configuration.h"
#include "Numerics/Quadrature.h"
#include "Particle/ParticleSet.h"
#include "QMCHamiltonians/ECPComponentBuilder.h"
#include "QMCHamiltonians/NonLocalECPotential.h"
#include "TestListenerFunction.h"

namespace qmcplusplus
{

TEST_CASE("NonLocalECPotential" "")
{
  using Real = QMCTraits::RealType;
  using FullPrecReal = QMCTraits::FullPrecRealType;
  using Position = QMCTraits::PosType;
  using testing::getParticularListener;

  CrystalLattice<OHMMS_PRECISION, OHMMS_DIM> lattice;
  lattice.BoxBConds = true; // periodic
  lattice.R.diagonal(1.0);
  lattice.reset();

  const SimulationCell simulation_cell(lattice);

  ParticleSet ions(simulation_cell);

  ions.setName("ion");
  ions.create({1});
  ions.R[0][0] = 0.0;
  ions.R[0][1] = 0.0;
  ions.R[0][2] = 0.0;

  SpeciesSet& ion_species       = ions.getSpeciesSet();
  int pIdx                      = ion_species.addSpecies("C");
  int pChargeIdx                = ion_species.addAttribute("charge");
  ion_species(pChargeIdx, pIdx) = 2;
  ions.createSK();
  ions.update();

  ParticleSet ions2(ions);
  ions2.update();

  ParticleSet elec(simulation_cell);
  elec.setName("elec");
  elec.create({1,1});
  elec.R[0][0] = 0.5;
  elec.R[0][1] = 0.0;
  elec.R[0][2] = 0.0;

  SpeciesSet& tspecies       = elec.getSpeciesSet();
  int upIdx                  = tspecies.addSpecies("u");
  int chargeIdx              = tspecies.addAttribute("charge");
  int massIdx                = tspecies.addAttribute("mass");
  tspecies(chargeIdx, upIdx) = -1;
  tspecies(massIdx, upIdx)   = 1.0;

  int dnIdx                  = tspecies.addSpecies("d");
  chargeIdx              = tspecies.addAttribute("charge");
  massIdx                = tspecies.addAttribute("mass");
  tspecies(chargeIdx, dnIdx) = -1;
  tspecies(massIdx, dnIdx)   = 1.0;

  elec.resetGroups();
  elec.createSK();
  const int ei_table_index = elec.addTable(ions);
  elec.update();

  ParticleSet elec2(elec);

  elec2.R[0][0] = 0.0;
  elec2.R[0][1] = 0.5;
  elec2.R[0][2] = 0.1;
  elec2.R[1][0] = 0.6;
  elec2.R[1][1] = 0.05;
  elec2.R[1][2] = -0.1;
  elec2.update();

  RefVector<ParticleSet> ptcls{elec, elec2};
  RefVectorWithLeader<ParticleSet> p_list(elec, ptcls);
  
  TrialWaveFunction psi, psi2;
  RefVectorWithLeader<TrialWaveFunction> twf_list(psi, {psi, psi2});

  bool doForces = false;
  bool use_DLA = false;
  
  NonLocalECPotential nl_ecp(ions, elec, psi, doForces, use_DLA);

  Matrix<Real> local_pots(2);
  Matrix<Real> local_pots2(2);

  ResourceCollection pset_res("test_pset_res");
  elec.createResource(pset_res);
  ResourceCollectionTeamLock<ParticleSet> pset_lock(pset_res, p_list);

  std::vector<ListenerVector<Real>> listeners;
  listeners.emplace_back("localpotential", getParticularListener(local_pots));
  listeners.emplace_back("localpotential", getParticularListener(local_pots2));

  Matrix<Real> ion_pots(2);
  Matrix<Real> ion_pots2(2);

  std::vector<ListenerVector<Real>> ion_listeners;
  ion_listeners.emplace_back("localpotential", getParticularListener(ion_pots));
  ion_listeners.emplace_back("localpotential", getParticularListener(ion_pots2));


  // This took some time to sort out from the multistage mess of put and clones 
  // but this accomplishes in a straight forward way what I interpret to be done by that code.
  Communicate* comm = OHMMS::Controller;
  ECPComponentBuilder ecp_comp_builder("test_read_ecp", comm, 4, 1);

  bool okay = ecp_comp_builder.read_pp_file("C.BFD.xml");
  REQUIRE(okay);
  UPtr<NonLocalECPComponent> nl_ecp_comp = std::move(ecp_comp_builder.pp_nonloc);
  nl_ecp_comp->initVirtualParticle(elec);
  nl_ecp.addComponent(0, std::move(nl_ecp_comp));
  UPtr<OperatorBase> nl_ecp2_ptr = nl_ecp.makeClone(elec2, psi2);
  auto& nl_ecp2 = *nl_ecp2_ptr;

  StdRandom<FullPrecReal> rng(10101);
  StdRandom<FullPrecReal> rng2(10201);
  nl_ecp.setRandomGenerator(&rng);
  nl_ecp2.setRandomGenerator(&rng2);
  
  RefVector<OperatorBase> nl_ecps{nl_ecp, nl_ecp2};
  RefVectorWithLeader<OperatorBase> o_list(nl_ecp, nl_ecps);
  ResourceCollection nl_ecp_res("test_nl_ecp_res");
  nl_ecp.createResource(nl_ecp_res);
  ResourceCollectionTeamLock<OperatorBase> nl_ecp_lock(nl_ecp_res, o_list);
  
  nl_ecp.mw_evaluatePerParticleWithToperator(o_list, twf_list, p_list, listeners, ion_listeners);

  CHECK(std::accumulate(local_pots.begin(), local_pots.begin() + local_pots.cols(), 0.0) == Approx(14.4822673798));
  CHECK(std::accumulate(local_pots2.begin(), local_pots2.begin() + local_pots2.cols(), 0.0) == Approx(14.4822673798));
  CHECK(std::accumulate(ion_pots.begin(), ion_pots.begin() + ion_pots.cols(), 0.0) == Approx(14.4822673798));
  CHECK(std::accumulate(ion_pots2.begin(), ion_pots2.begin() + ion_pots2.cols(), 0.0) == Approx(14.4822673798));

  elec.R[0][0] = 0.5;
  elec.R[0][1] = 0.0;
  elec.R[0][2] = 2.0;
  elec.update();
  
  nl_ecp.mw_evaluatePerParticleWithToperator(o_list, twf_list, p_list, listeners, ion_listeners);

  CHECK(std::accumulate(local_pots.begin(), local_pots.begin() + local_pots.cols(), 0.0) == Approx(14.4822673798));

}

}
