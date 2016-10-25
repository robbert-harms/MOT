Maastricht Optimization Toolbox
===============================

.. image:: https://badge.fury.io/py/mot.png
    :target: http://badge.fury.io/py/mot


A library for parallel optimization and sampling using Python and OpenCL.

* Free software: LGPL v3 license
* Full documentation: https://mot.readthedocs.org
* Project home: https://github.com/cbclab/MOT
* Uses the `GitLab workflow <https://docs.gitlab.com/ee/workflow/gitlab_flow.html>`_
* Tags: optimization, parallel, opencl, python

For the diffusion MRI package MDT which builds on top of this, please see https://github.com/cbclab/MDT.


Quick installation guide
^^^^^^^^^^^^^^^^^^^^^^^^
The basic requirements for MOT are:

* Python 3.x (recommended) or Python 2.7
* OpenCL 1.2 (or higher) supper in GPU driver or CPU runtime


**Linux**

For Ubuntu >= 16 you can use:

* ``sudo add-apt-repository ppa:robbert-harms/cbclab``
* ``sudo apt-get update``
* ``sudo apt-get install python3-mot``


For Debian users and Ubuntu < 16 users, install MOT with:

* ``sudo apt-get install python3 python3-pip python3-pyopencl``
* ``sudo pip3 install MOT``


**Windows**

* Install Python Anaconda 3.* 64bit from https://www.continuum.io/downloads
* Compile PyOpenCL with ``pip install pyopencl``, this requires:
    * Visual Studio 2015 (Community edition or higher) with the Python and Common Tools for Visual C++ options enabled
    * OpenCL development kit (`NVidia CUDA <https://developer.nvidia.com/cuda-downloads>`_ or `Intel OpenCL SDK <https://software.intel.com/en-us/intel-opencl>`_ or the `AMD APP SDK <http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/>`_)
* Open a Anaconda shell and type: ``pip install MOT``


For more information and for more elaborate installation instructions, please see: https://mot.readthedocs.org


Caveats
^^^^^^^
There are a few caveats and known issues, primarily related to OpenCL:

* Windows support is experimental due to the difficulty of installing PyOpenCL, hopefully installing PyOpenCL will get easier on Windows soon.
* For AMD users with Ubuntu >= 16, the new AMD GPU-Pro driver is still in beta and may not work with all the kernels in MOT.
  Our recommendation at the moment (October 2016) is to use Ubuntu version 14.
* GPU acceleration is not possible in most virtual machines due to lack of GPU or PCI-E pass-through, this will change whenever virtual machines vendors program this feature.
  Our recommendation is to install neurodebian on your machine and run it on that.
