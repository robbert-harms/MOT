Installation
************
.. highlight:: console

The installation is a three step procedure:

1. Installing a :ref:`Python interpreter <install_python>`
2. Installing the :ref:`PyOpenCL drivers and Python bindings <install_opencl>`
3. :ref:`Install MOT <install_mot>`


.. _install_python:

Installing Python
=================
Since it is a Python package we need to install the Python interpreter. Considering that Python2 is soon end of life, this package focuses on installing it using Python3.

Linux (Ubuntu)
--------------

.. code-block:: bash

    $ apt-get install python3 python3-pip


Windows
-------
The easiest way is with Anaconda. Install the Python3.x bindings from `Anaconda <https://www.continuum.io/downloads>`_.


.. _install_opencl:

Installing OpenCL
=================
For OpenCL you need two things, an OpenCL driver for your platform and the Python OpenCL bindings.

Linux (Ubuntu)
--------------
On Ubuntu, the easiest way to install all of this:

.. code-block:: bash

    $ apt-get install python3-pyopencl


Windows
-------
On Windows, make sure you install the correct OpenCL driver (Intel/AMD/NVidia). For graphics cards the drivers are normally already installed. After that, Anaconda should automatically install the Python bindings.


.. _install_mot:

Installing MOT
==============
With OpenCL and Python installed you can now install MOT.

Linux (Ubuntu)
--------------

.. code-block:: bash

    $ pip3 install MOT

To test the installation, please open a Python shell and try to import MOT.


Windows
-------
Open an Anaconda shell and use:

.. code-block:: bash

    $ pip install MOT


Testing the installation
========================

Open a python shell (using Anaconda in Windows and bash in Linux) and type:

.. code-block:: python

    >>> import mot
    >>> from mot.cl_environments import CLEnvironmentFactory
    >>> CLEnvironmentFactory.smart_device_selection()

If you got no errors and the output is a list of CL environments the installation was successful.

