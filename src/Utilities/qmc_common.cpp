//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Anouar Benali, benali@anl.gov, Argonne National Laboratory
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Paul R. C. Kent, kentpr@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "qmc_common.h"
#include "config.h"
#include "qmcpack_version.h"
#include "Message/Communicate.h"
#include "Platforms/Host/OutputManager.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace qmcplusplus
{
QMCState::QMCState()
{
  is_restart             = false;
  use_density            = false;
  dryrun                 = false;
  io_node                = true;
  mpi_groups             = 1;
  qmc_counter            = 0;
  memory_allocated       = 0;
}

void QMCState::initialize(int argc, char** argv)
{
  io_node     = (OHMMS::Controller->rank() == 0);
  bool stopit = false;
  //going to use better option library
  int i = 1;
  while (i < argc)
  {
    std::string c(argv[i]);
    if (c.find("--dryrun") < c.size())
    {
      dryrun = true;
    }
    else if (c.find("--save_wfs") < c.size())
    {
      if (io_node)
        std::cerr << std::endl
                  << "ERROR: command line option --save_wfs has been removed."
                  << "Use save_coefs input tag as described in the manual." << std::endl;
      stopit = true;
    }
    else if (c.find("--help") < c.size())
    {
      stopit = true;
    }
    else if (c.find("--version") < c.size())
    {
      stopit = true;
    }
    else if (c.find("--vacuum") < c.size())
    {
      if (io_node)
        std::cerr << std::endl
                  << "ERROR: command line option --vacuum has been removed. "
                  << "Use vacuum input tag as described in the manual." << std::endl;
      stopit = true;
    }
    else if (c.find("--noprint") < c.size())
    { //do not print Jastrow or PP
      io_node = false;
    }
    ++i;
  }
  if (stopit && io_node)
  {
    std::cerr << std::endl
              << "QMCPACK version " << QMCPACK_VERSION_MAJOR << "." << QMCPACK_VERSION_MINOR << "."
              << QMCPACK_VERSION_PATCH << " built on " << __DATE__ << std::endl;
    print_git_info_if_present(std::cerr);
    std::cerr << std::endl << "Usage: qmcpack input [--dryrun --gpu]" << std::endl << std::endl;
  }
  if (stopit)
  {
    OHMMS::Controller->finalize();
    exit(1);
  }
}

void QMCState::print_options(std::ostream& os)
{
  os << "  Global options " << std::endl;
  if (dryrun)
    os << "  dryrun : qmc sections will be ignored." << std::endl;
}

void QMCState::print_memory_change(const std::string& who, size_t before)
{
  before = memory_allocated - before;
  app_log() << "MEMORY increase " << (before >> 20) << " MB " << who << std::endl;
}

void QMCState::print_git_info_if_present(std::ostream& os)
{
#ifdef QMCPACK_GIT_BRANCH
  os << std::endl;
  os << "  Git branch: " << QMCPACK_GIT_BRANCH << std::endl;
  os << "  Last git commit: " << QMCPACK_GIT_HASH << std::endl;
  os << "  Last git commit date: " << QMCPACK_GIT_COMMIT_LAST_CHANGED << std::endl;
  os << "  Last git commit subject: " << QMCPACK_GIT_COMMIT_SUBJECT << std::endl;
#endif
}

void QMCState::print_time_per_rank(const std::string& name, double t)
{
  static std::ofstream *fout = nullptr;
  static int seq_id = 0;
  if(fout == nullptr)
  {
    int rank = OHMMS::Controller->rank();

    // Get the name of the processor
    char processorName[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    MPI_Get_processor_name(processorName, &nameLen);

    char fname[128];
    //const char* value = std::getenv("QMCPACK_OUT");
    //std::string out_dir{std::getenv("PWD")};
    //if (value != nullptr) {
    //  out_dir = value;
    //}
    //sprintf(fname,"%s/%08d.%s.txt", out_dir.c_str(), rank, processorName);
    sprintf(fname,"%08d.%s.txt", rank, processorName);
    fout = new std::ofstream(fname);
  }

  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  std::tm now_tm = *std::localtime(&now_c);
  auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000;
  std::stringstream ss;
  ss << std::put_time(&now_tm, "%H:%M:%S");
  ss << '.' << std::setfill('0') << std::setw(6) << microseconds.count();

  *fout << ss.str() << " " << seq_id << "_" << name << " " << t << std::endl;
  seq_id++;
}

QMCState qmc_common;

} // namespace qmcplusplus
