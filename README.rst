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


**Linux (Ubuntu)**

* ``sudo apt-get install python3 python3-pip python3-pyopencl``
* ``sudo pip3 install MOT``


**Windows**

* Install Python Anaconda 3.* 64bit from https://www.continuum.io/downloads
* Install PyOpenCL:
    * Using a binary package from http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl (preferred method)
    * Compile PyOpenCL with ``pip install pyopencl``, this requires:
        * Visual Studio 2015 (Community edition or higher) with the Python and Common Tools for Visual C++ options enabled
        * OpenCL development kit (NVidia CUDA or Intel OpenCL SDK or the AMD APP SDK)
* Open a Anaconda shell and type: ``pip install MOT``


For more information and for more elaborate installation instructions, please see: https://mot.readthedocs.org
