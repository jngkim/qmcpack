//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Cynthia Gu, zg1@ornl.gov, Oak Ridge National Laboratory
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/**
 * @file QMCDriver.h
 * @brief Declaration of QMCDriver
 */
#ifndef QMCPLUSPLUS_QMCDRIVER_H
#define QMCPLUSPLUS_QMCDRIVER_H

#include "Configuration.h"
#include "OhmmsData/ParameterSet.h"
#include "Pools/PooledData.h"
#include "Utilities/TimerManager.h"
#include "Utilities/ScopedProfiler.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/WaveFunctionPool.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "Estimators/EstimatorManagerBase.h"
#include "QMCDrivers/DriverTraits.h"
#include "QMCDrivers/QMCDriverInterface.h"
#include "QMCDrivers/GreenFunctionModifiers/DriftModifierBase.h"
#include "QMCDrivers/SimpleFixedNodeBranch.h"
#include "QMCDrivers/BranchIO.h"
class Communicate;

namespace qmcplusplus
{
/** @defgroup QMCDrivers QMC Driver group
 * QMC drivers that implement QMC algorithms
 */

/** @defgroup WalkerByWalker QMC Drivers using walker-by-walker update
 * @brief Move all the particles for each walker
 */

/** @defgroup ParticleByParticle QMC Drivers using particle-by-particle update
 * @brief Move particle by particle
 */

/** @defgroup MultiplePsi QMC Drivers for energy differences
 * @brief Umbrella sampling over multiple H/Psi
 *
 * This class of QMC drivers are suitable to evaluate
 * the energy differences of multiple H-Psi pairs.
 */

//forward declarations: Do not include headers if not needed
class MCWalkerConfiguration;
class HDFWalkerOutput;
class TraceManager;

/** @ingroup QMCDrivers
 * @{
 * @brief abstract base class for QMC engines
 */
class QMCDriver : public QMCDriverInterface, public QMCTraits, public MPIObjectBase
{
public:
  /** enumeration coupled with QMCMode */
  enum
  {
    QMC_UPDATE_MODE,
    QMC_MULTIPLE,
    QMC_OPTIMIZE,
    QMC_WARMUP
  };

  typedef MCWalkerConfiguration::Walker_t Walker_t;
  typedef Walker_t::Buffer_t Buffer_t;
  /** bits to classify QMCDriver
   *
   * - qmc_driver_mode[QMC_UPDATE_MODE]? particle-by-particle: walker-by-walker
   * - qmc_driver_mode[QMC_MULTIPLE]? multiple H/Psi : single H/Psi
   * - qmc_driver_mode[QMC_OPTIMIZE]? optimization : vmc/dmc/rmc
   */
  std::bitset<QMC_MODE_MAX> qmc_driver_mode;

  /// whether to allow traces
  bool allow_traces;
  /// traces xml
  xmlNodePtr traces_xml;

  /// Constructor.
  QMCDriver(MCWalkerConfiguration& w,
            TrialWaveFunction& psi,
            QMCHamiltonian& h,
            Communicate* comm,
            const std::string& QMC_driver_type,
            bool enable_profiling = false);

  ///Copy Constructor (disabled).
  QMCDriver(const QMCDriver&) = delete;
  ///Copy operator (disabled).
  QMCDriver& operator=(const QMCDriver&) = delete;

  ~QMCDriver() override;

  ///return current step
  inline int current() const { return CurrentStep; }

  /** set the update mode
   * @param pbyp if true, use particle-by-particle update
   */
  inline void setUpdateMode(bool pbyp) override { qmc_driver_mode[QMC_UPDATE_MODE] = pbyp; }

  /** Set the status of the QMCDriver
   * @param aname the root file name
   * @param h5name root name of the master hdf5 file containing previous qmcrun
   * @param append if true, the run is a continuation of the previous qmc
   *
   * All output files will be of
   * the form "aname.s00X.suffix", where "X" is number
   * of previous QMC runs for the simulation and "suffix"
   * is the suffix for the output file.
   */
  void setStatus(const std::string& aname, const std::string& h5name, bool append) override;

  /** add QMCHamiltonian/TrialWaveFunction pair for multiple
   * @param h QMCHamiltonian
   * @param psi TrialWaveFunction
   *
   * *Multiple* drivers use multiple H/Psi pairs to perform correlated sampling
   * for energy difference evaluations.
   */
  void add_H_and_Psi(QMCHamiltonian* h, TrialWaveFunction* psi) override;

  /** initialize with xmlNode
   */
  void process(xmlNodePtr cur) override;

  /** return a xmlnode with update **/
  xmlNodePtr getQMCNode();

  void putWalkers(std::vector<xmlNodePtr>& wset) override;

  inline void putTraces(xmlNodePtr txml) override { traces_xml = txml; }

  inline void requestTraces(bool traces) override { allow_traces = traces; }

  std::string getEngineName() override { return QMCType; }

  template<class PDT>
  void setValue(const std::string& aname, PDT x)
  {
    m_param.setValue(aname, x);
  }

  ///set the BranchEngineType
  void setBranchEngine(std::unique_ptr<BranchEngineType>&& be) override { branchEngine = std::move(be); }

  ///return BranchEngineType*
  std::unique_ptr<BranchEngineType> getBranchEngine() override { return std::move(branchEngine); }

  int addObservable(const std::string& aname)
  {
    if (Estimators)
      return Estimators->addObservable(aname.c_str());
    else
      return -1;
  }

  RealType getObservable(int i) { return Estimators->getObservable(i); }

  void setTau(RealType i) { Tau = i; }

  ///set global offsets of the walkers
  void setWalkerOffsets();

  ///Observables manager
  EstimatorManagerBase* Estimators;

  ///Traces manager
  std::unique_ptr<TraceManager> Traces;

  ///return the random generators
  inline RefVector<RandomGenerator_t> getRngRefs() const
  {
    RefVector<RandomGenerator_t> RngRefs;
    for (int i = 0; i < Rng.size(); ++i)
      RngRefs.push_back(*Rng[i]);
    return RngRefs;
  }

  ///return the i-th random generator
  inline RandomGenerator_t& getRng(int i) override { return (*Rng[i]); }

  unsigned long getDriverMode() override { return qmc_driver_mode.to_ulong(); }

protected:
  ///branch engine
  std::unique_ptr<BranchEngineType> branchEngine;
  ///drift modifer
  DriftModifierBase* DriftModifier;
  ///randomize it
  bool ResetRandom;
  ///flag to append or restart the run
  bool AppendRun;
  ///flag to turn off dumping configurations
  bool DumpConfig;
  ///true, if it is a real QMC engine
  bool IsQMCDriver;
  /** the number of times this QMCDriver is executed
   *
   * MyCounter is initialized to zero by the constructor and is incremented
   * whenever a run is completed by calling finalize(int block) or
   * using MyCounter++ as in RQMC.
   */
  int MyCounter;
  ///the number to delay updates by
  int kDelay;
  /** period of dumping walker configurations and everything else for restart
   *
   * The unit is a block.
   */
  int Period4CheckPoint;
  /** period of dumping walker positions and IDs for Forward Walking
  *
  * The unit is in steps.
  */
  int storeConfigs;

  ///Period to recalculate the walker properties from scratch.
  int Period4CheckProperties;

  /** period of recording walker configurations
   *
   * Default is 0 indicating that only the last configuration will be saved.
   */
  int Period4WalkerDump;

  /** period of recording walker positions and IDs for forward walking afterwards
   *
   */
  int Period4ConfigDump;

  ///current step
  IndexType CurrentStep;

  ///maximum number of blocks
  IndexType nBlocks;

  ///maximum number of steps
  IndexType nSteps;

  ///number of steps between a step: VMCs do not evaluate energies
  IndexType nSubSteps;

  ///number of warmup steps
  IndexType nWarmupSteps;

  ///counter for number of moves accepted
  IndexType nAccept;

  ///counter for number of moves /rejected
  IndexType nReject;

  /// the number of blocks between recomptePsi
  IndexType nBlocksBetweenRecompute;

  ///the number of walkers
  IndexType nTargetWalkers;
  ///the number of saved samples
  IndexType nTargetSamples;
  ///alternate method of setting QMC run parameters
  IndexType nStepsBetweenSamples;
  ///samples per thread
  RealType nSamplesPerThread;
  ///target population
  RealType nTargetPopulation;


  ///timestep
  RealType Tau;

  ///maximum cpu in secs
  int MaxCPUSecs;

  ///Time-step factor \f$ 1/(2\tau)\f$
  RealType m_oneover2tau;
  ///Time-step factor \f$ \sqrt{\tau}\f$
  RealType m_sqrttau;

  ///pointer to qmc node in xml file
  xmlNodePtr qmcNode;

  ///type of QMC driver
  const std::string QMCType;
  ///the root of h5File
  std::string h5FileRoot;
  ///root of all the output files
  std::string RootName;

  ///store any parameter that has to be read from a file
  ParameterSet m_param;

  ///walker ensemble
  MCWalkerConfiguration& W;

  ///trial function
  TrialWaveFunction& Psi;

  ///Hamiltonian
  QMCHamiltonian& H;

  ///record engine for walkers
  std::unique_ptr<HDFWalkerOutput> wOut;

  ///a list of TrialWaveFunctions for multiple method
  std::vector<TrialWaveFunction*> Psi1;

  ///a list of QMCHamiltonians for multiple method
  std::vector<QMCHamiltonian*> H1;

  ///Random number generators
  UPtrVector<RandomGenerator_t> Rng;

  ///a list of mcwalkerset element
  std::vector<xmlNodePtr> mcwalkerNodePtr;

  ///temporary storage for drift
  ParticleSet::ParticlePos_t drift;

  ///temporary storage for random displacement
  ParticleSet::ParticlePos_t deltaR;

  ///spin mass for spinor calcs
  RealType SpinMass;

  bool putQMCInfo(xmlNodePtr cur);

  void addWalkers(int nwalkers);

  /** record the state of the block
   * @param block current block
   *
   * virtual function with a default implementation
   */
  void recordBlock(int block) override;

  /** finalize a qmc section
   * @param block current block
   * @param dumpwalkers if true, dump walkers
   *
   * Accumulate energy and weight is written to a hdf5 file.
   * Finialize the estimators
   */
  bool finalize(int block, bool dumpwalkers = true);

  int rotation;
  std::string getRotationName(std::string RootName);
  std::string getLastRotationName(std::string RootName);
  const std::string& get_root_name() const override { return RootName; }

private:
  NewTimer* checkpointTimer;
  ///time the driver lifetime
  ScopedTimer driver_scope_timer_;
  ///profile the driver lifetime
  ScopedProfiler driver_scope_profiler_;
};
/**@}*/
} // namespace qmcplusplus

#endif
