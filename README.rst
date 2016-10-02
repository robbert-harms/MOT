===============================
Maastricht Optimization Toolbox
===============================

.. image:: https://badge.fury.io/py/mot.png
    :target: http://badge.fury.io/py/mot


A library for parallel optimization and sampling in python and opencl.

* Free software: LGPL v3 license
* Full documentation: https://mot.readthedocs.org
* Project home: https://github.com/robbert-harms/MOT
* Uses the `GitLab workflow <https://docs.gitlab.com/ee/workflow/gitlab_flow.html>`_
* Tags: optimization, parallel, opencl, python


Installation
------------
.. highlight:: console

The installation is a three step procedure:

1. Installing a Python interpreter
2. Installing the PyOpenCL drivers and Python bindings
3. Install MOT


|
Installing Python
^^^^^^^^^^^^^^^^^
Since it is a Python package we need to install the Python interpreter. Considering that Python2 is soon end of life, this package focuses on installing it using Python3.

Linux (Ubuntu)
""""""""""""""
``apt-get install python3 python3-pip``


Windows
"""""""
The easiest way is with Anaconda. Install the Python3.x bindings from `Anaconda <https://www.continuum.io/downloads>`_.

|
Installing OpenCL bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^
For OpenCL you need two things, an OpenCL driver for your platform and the Python OpenCL bindings.

Linux (Ubuntu)
""""""""""""""
On Ubuntu, the easiest way to install all of this:

``apt-get install python3-pyopencl``


Windows
"""""""
On Windows, make sure you install the correct OpenCL driver (Intel/AMD/NVidia). For graphics cards the drivers are normally already installed. After that, Anaconda should automatically install the Python bindings.


|
Installing MOT
^^^^^^^^^^^^^^
With OpenCL and Python installed you can now install MOT.

Linux (Ubuntu)
""""""""""""""
``pip3 install MOT``


Windows
"""""""
Open an Anaconda shell and use:

``pip install MOT``
