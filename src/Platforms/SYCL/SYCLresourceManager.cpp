//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp 
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp
////////////////////////////////////////////////////////////////////////////////////// // -*- C++ -*-
#include <Utilities/Communicate.h>
#include <CL/sycl.hpp>
namespace qmcplusplus
{
  /** create a list of gpu devices 
   * @param t_platorm name of Platform: level_zero, opencl, CUDA
   */
  inline std::vector<sycl::device> get_devices(const char* t_platform)
  {
    std::vector<sycl::device> gpu;

    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    bool look_for_it=true;
    unsigned i=0;
    while(look_for_it && i<platforms.size())
    {
      auto p_name = platforms[i].get_info<sycl::info::platform::name>();
      if(p_name.find(t_platform)<p_name.size())
      {
        look_for_it=false; // pick only the first valid platform
        std::vector<sycl::device> one_platform_devices = platforms[i].get_devices();
        for(auto& dev: one_platform_devices)
        {
          if(dev.is_gpu())
          {
            gpu.push_back(dev);
          }
        }
      }
      ++i;
    }

    return gpu;
  }

  /** Create a list of devices for the applications managed by a MPI processor
   *
   * This should be handled by MPI runtime.
   * @param mpi_rank
   * @param mpi_size
   * @param device_fission if true, tiles will be used as a device.
   */
  inline std::vector<sycl::device> get_devices(int mpi_rank, int mpi_size, bool device_fission=true)
  {
#if defined(__INTEL_LLVM_COMPILER)
    std::string rt_tag{"evel"};
    if(const char* env_p=std::getenv("SYCL_DEVICE_FILTER"))
    {
      std::string rt_tag_env{env_p};
      if(rt_tag_env.find("opencl")<rt_tag_env.size())
        rt_tag="opencl";
    }

    std::vector<sycl::device> gpus=get_devices(rt_tag.c_str());
    std::vector<sycl::device> all_devices;
    if(device_fission)
    {
      for(auto& g : gpus)
      {
        try
        {
          std::vector<sycl::device> tiles=
            g.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
                sycl::info::partition_affinity_domain::numa);
          all_devices.insert(all_devices.end(),tiles.begin(),tiles.end());
        }
        catch(sycl::feature_not_supported)
        {
          all_devices.push_back(g);
        }
      }
    }
    else
    {
      all_devices=gpus;
    }

    if(mpi_size>all_devices.size())
    {
      return std::vector<sycl::device>{all_devices[mpi_rank%all_devices.size()]};
    }
    else
    {
      int ndev_rank=all_devices.size()/mpi_size;
      // a subset of devices (compact distribution)
      return std::vector<sycl::device>{all_devices.begin()+mpi_rank*ndev_rank,
        all_devices.begin()+(mpi_rank+1)*ndev_rank};
    }
#else
    //cannot create context with multiple devices
    std::vector<sycl::device> gpus=get_devices("CUDA");
    return std::vector<sycl::device>{gpus[mpi_rank%gpus.size()]};
#endif
  }

  struct SyclResourceManager
  {
    ///devices per rank
    std::vector<sycl::device> mDevices;
    ///main context
    std::unique_ptr<sycl::context> mContext;
    ///main context
    std::unique_ptr<sycl::queue> mQueue;

    SyclResourceManager()=delete;
    SyclResourceManager(const SyclResourceManager& )=delete;
    SyclResourceManager(SyclResourceManager&& )=delete;
    ~SyclResourceManager()=default;

    explicit SyclResourceManager(int mpi_rank, int mpi_size)
    {
      mDevices=get_devices(mpi_rank,mpi_size);
      mContext=std::unique_ptr<sycl::context>(new sycl::context{mDevices});
      //default queue
      mQueue=std::unique_ptr<sycl::queue>(new sycl::queue{*mContext.get(),mDevices[0]});
    }

    inline sycl::queue create_queue(int ip)
    {
      if(ip<0)
        return *mQueue;
      else
        return sycl::queue{*mContext.get(),mDevices[ip%mDevices.size()]};
    }
  };

  /** singleton to manage SYCL resources */
  sycl::queue get_default_queue(int ip)
  {
    static std::unique_ptr<SyclResourceManager> sycl_manager;

    if(sycl_manager.get()==nullptr)
    {
      int m_size{1}, m_rank{0};
#ifdef HAVE_MPI
      int iflag;
      MPI_Initialized(&iflag);
      if(!iflag)
      {//only for unit tests
        int argc=0;
        char** argv=0;
        MPI_Init(&argc, &argv);
      }
      MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &m_size);
#endif
      sycl_manager=std::unique_ptr<SyclResourceManager>(new SyclResourceManager(m_rank,m_size));
    }
    return sycl_manager->create_queue(ip);
  }
}
