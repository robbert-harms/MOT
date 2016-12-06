###############################
Maastricht Optimization Toolbox
###############################
The Maastricht Optimization Toolbox, MOT, is a library for parallel optimization and sampling using the graphics card for the computations.
It is meant to optimize, in parallel, a large number of smaller problems, in contrast to optimizing one big problem with parallelized parts.
For example, in diffusion MRI the brain is scanned in a 3D grid where each grid element, a *voxel*, represents its own optimization problem.
The number of data points per voxel is generally small, ranging from 30 to 500 datapoints, and the models fitted to that data have
generally somewhere between 6 and 20 parameters.
Since each of these voxels can be analyzed independently of the others, the computations can be massively parallelized and hence programming
for the graphics card can allow for a large speed gain.
This software toolbox was originally built for exactly this use case, yet the algorithms and data structures are generalized such that any scientific field may take advantage of this toolbox.
For the diffusion MRI package *MDT* to which is referred in this example, please see https://github.com/cbclab/MDT.


****************
Can MOT help me?
****************
To recognize if MOT can help you with your use case, try to see if your computations can be parallized in some way.
If you have just one big optimization problem with 10.000 variables, MOT unfortunately can not help you.
On the other hand, if you find a way to split your analysis in (a lot of; >10.000) smaller sub-problems, with ~30 parameters or less each, MOT may actually be of help.


*******
Summary
*******
* Free software: LGPL v3 license
* Interface in Python, computations in OpenCL
* Full documentation: https://mot.readthedocs.org
* Project home: https://github.com/cbclab/MOT
* PyPi package: `PyPi <http://badge.fury.io/py/mot>`_
* Uses the `GitLab workflow <https://docs.gitlab.com/ee/workflow/gitlab_flow.html>`_
* Tags: optimization, parallel, opencl, python


************************
Quick installation guide
************************
The basic requirements for MOT are:

* Python 3.x (recommended) or Python 2.7
* OpenCL 1.2 (or higher) support in GPU driver or CPU runtime


**Linux**

For Ubuntu >= 16 you can use:

* ``sudo add-apt-repository ppa:robbert-harms/cbclab``
* ``sudo apt update``
* ``sudo apt install python3-mot``


For Debian users and Ubuntu < 16 users, install MOT with:

* ``sudo apt install python3 python3-pip python3-pyopencl python3-devel``
* ``sudo pip3 install mot``


**Windows**
For Windows the short guide is:

* Install Python Anaconda 3.* 64bit from https://www.continuum.io/downloads
* Install or upgrade your GPU drivers
* Install PyOpenCL using one of the following methods:
    1. Use a binary, for example from http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl or;
    2. Compile PyOpenCL with ``pip install pyopencl``, this requires:
        * Visual Studio 2015 (Community edition or higher) with the Python and Common Tools for Visual C++ options enabled
        * OpenCL development kit (`NVidia CUDA <https://developer.nvidia.com/cuda-downloads>`_ or `Intel OpenCL SDK <https://software.intel.com/en-us/intel-opencl>`_ or the `AMD APP SDK <http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/>`_)
* Open a Anaconda shell and type: ``pip install mot``


For more information and for more elaborate installation instructions, please see: https://mot.readthedocs.org


*******
Caveats
*******
There are a few caveats and known issues, primarily related to OpenCL:

* Windows support is experimental due to the difficulty of installing PyOpenCL, hopefully installing PyOpenCL will get easier on Windows soon.
* For AMD users with Ubuntu >= 16, the new AMD GPU-Pro driver is still in beta and may not work with all the optimalization routines    cd  in MOT.
  Our recommendation at the moment (October 2016) is to use Ubuntu version 14 when using AMD hardware.
* GPU acceleration is not possible in most virtual machines due to lack of GPU or PCI-E pass-through, this will change whenever virtual machines vendors program this feature.
  Our recommendation is to install Linux on your machine directly.
