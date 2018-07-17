###################################
Multi-threaded Optimization Toolbox
###################################
The Multi-threaded Optimization Toolbox (MOT) is a library for parallel optimization and sampling using the OpenCL compute platform.
Using OpenCL allows parallel processing using all CPU cores or using the GPU (Graphics card).
MOT implements OpenCL parallelized versions of the Powell, Nelder-Mead Simplex and Levenberg-Marquardt non-linear optimization algorithms
alongside various flavors of Markov Chain Monte Carlo (MCMC) sampling.

For the full documentation see: https://mot.readthedocs.org


****************
Can MOT help me?
****************
MOT can help you if you have multiple small independent optimization problems.
For example, if you have a lot of (>10.000) small optimization problems, with ~30 parameters or less each, MOT may be of help.
If, on the other hand, you have one big optimization problem with 10.000 variables, MOT unfortunately can not help you.


****************
Example use case
****************
MOT was originally written as a computation package for the `Microstructure Diffusion Toolbox <https://github.com/cbclab/MDT>`_, used in dMRI brain research.
In diffusion Magnetic Resonance Imaging (dMRI) the brain is scanned in a 3D grid where each grid element, a *voxel*, represents its own optimization problem.
The number of data points per voxel is generally small, ranging from 30 to 500 datapoints, and the models fitted to that data have generally
somewhere between 6 and 20 parameters.
Since each of these voxels can be analyzed independently of the others, the computations can be massively parallelized and hence programming
in OpenCL potentially allows large speed gains.
This software toolbox was originally built for exactly this use case, yet the algorithms and data structures are generalized such that any
scientific field may take advantage of this toolbox.

For the diffusion MRI package *MDT* to which is referred in this example, please see https://github.com/cbclab/MDT.


*******
Summary
*******
* Free software: LGPL v3 license
* Interface in Python, computations in OpenCL
* Implements Powell, Nelder-Mead Simplex and Levenberg-Marquardt non-linear optimization algorithms
* Implements various Markov Chain Monte Carlo (MCMC) sampling routines
* Tags: optimization, sampling, parallel, opencl, python


*****
Links
*****
* Full documentation: https://mot.readthedocs.org
* Project home: https://github.com/cbclab/MOT
* PyPi package: `PyPi <http://badge.fury.io/py/mot>`_


************************
Quick installation guide
************************
The basic requirements for MOT are:

* Python 3.x
* OpenCL 1.2 (or higher) support in GPU driver or CPU runtime


**Linux**

For Ubuntu >= 16 you can use:

* ``sudo add-apt-repository ppa:robbert-harms/cbclab``
* ``sudo apt update``
* ``sudo apt install python3-pip python3-mot``
* ``sudo pip3 install tatsu``


For Debian users and Ubuntu < 16 users, install MOT with:

* ``sudo apt install python3 python3-pip python3-pyopencl python3-devel``
* ``sudo pip3 install mot``


**Mac**

* Install Python Anaconda 3.* 64bit from https://www.continuum.io/downloads>
* Open a terminal and type ``pip install mot``


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
* GPU acceleration is not possible in most virtual machines due to lack of GPU or PCI-E pass-through, this will change whenever virtual machines vendors program this feature.
  Our recommendation is to install Linux on your machine directly.
