===============================
Maastricht Optimization Toolbox
===============================

.. image:: https://badge.fury.io/py/mot.png
    :target: http://badge.fury.io/py/mot


A library for parallel optimization and sampling in python and opencl.

* Free software: LGPL v3 license
* Documentation: https://mot.readthedocs.org.

Installation
------------
Installing Python
^^^^^^^^^^^^^^^^^
Since it is a Python package we need to install the Python interpreter. Considering that Python2 is soon end of life, this package focuses on installing it using Python3. 

### Linux (Ubuntu)
```sh
apt-get install python3 python3-pip
```

### Windows
The easiest way is with Anaconda. Install the Python3.x bindings from [Anaconda](https://www.continuum.io/downloads)


Installing OpenCL bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^
For OpenCL you need two things, an OpenCL driver for your platform and the Python OpenCL bindings.

### Linux (Ubuntu)
On Ubuntu, the easiest way to install all of this:
```sh
apt-get install python3-pyopencl
```

### Windows
On Windows, make sure you install the correct OpenCL driver (Intel/AMD/NVidia). For graphics cards the drivers are normally already installed. After that, Anaconda should automatically install the Python bindings.


Installing MOT
^^^^^^^^^^^^^^
With OpenCL available you can now install MOT.

### Linux (Ubuntu)
Preferably use Python3 (Python2 will be deprecated at one point):
```sh
pip3 install MOT
```

If you insist on Python2, install it with
```sh
pip install MOT
```

### Windows
Open an Anaconda shell and use:
```sh
pip install MOT
```
