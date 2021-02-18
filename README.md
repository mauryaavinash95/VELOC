# VELOC: VEry-Low Overhead Checkpointing System

VeloC is a multi-level checkpoint/restart runtime that delivers 
high performance and scalability for complex heterogeneous storage 
hierarchies without sacrificing ease of use and flexibility.

It is primarily used as a fault-tolerance tool for tightly coupled
HPC applications running on supercomputing infrastructure but is
essential in many other use cases: suspend-resume, migration, 
debugging.

VeloC is a collaboration between Argonne National Laboratory and 
Lawrence Livermore National Laboratory as part of the Exascale 
Computing Project.

## VELOC-GPU
The VELOC-GPU project is an experimental prototype for enabling checkpoint/restart
support on NVidia GPUs. The parent VELOC source is available at: https://github.com/ECP-VeloC/VELOC
and the VELOC-GPU source is available at: https://github.com/ECP-VeloC/gpu-enabled.

To run VELOC with GPU support, perform the following steps:
1. git clone https://github.com/ECP-VeloC/VELOC.git
2. git clone https://github.com/ECP-VeloC/gpu-enabled.git  # Private repository
3. cp -r gpu-enabled/src/lib/* VELOC/src/lib/
4. cp gpu-enabled/CMakeLists.txt VELOC/CMakeLists.txt
5. cd gpu-enabled
6. ./bootstrap.sh
7. ./auto-install.py <install_dir>

## Documentation

The documentation of VeloC is available here: http://veloc.rtfd.io

It includes a quick start guide as well that covers the basics needed
to use VeloC on your system.

## Contacts

In case of questions and comments or help, please contact the VeloC team at 
veloc-users@lists.mcs.anl.gov


## Release

Copyright (c) 2018-2020, UChicago Argonne LLC, operator of Argonne National Laboratory <br>
Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.

For release details and restrictions, please read the [LICENSE](https://github.com/ECP-VeloC/VELOC/blob/master/LICENSE) 
and [NOTICE](https://github.com/ECP-VeloC/VELOC/blob/master/NOTICE) files.

`LLNL-CODE-751725` `OCEC-18-060`
