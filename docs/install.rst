Installation
************


Ubuntu / Debian Linux
=====================
Using the package manager installation in Ubuntu and Debian is easy. To install the dependencies please use:

.. code-block:: bash

    $ sudo apt-get install python3 python3-pip python3-pyopencl

To install *python3*, *pip3* and the *pyopencl* Python package.

Then, to install MOT, please type:

.. code-block:: bash

    $ sudo pip3 install MOT

Please continue with testing the installation at the end of this chapter.


Windows
=======
The installation on Windows is a little bit more convoluted due to the lack of a package manager. The installation is a three step procedure:

1. Installing a :ref:`Python interpreter <install_python>`
2. Installing the :ref:`PyOpenCL drivers and Python bindings <install_opencl>`
3. :ref:`Installing MOT <install_mot>`


.. _install_python:

Installing Python
-----------------
Since MOT is a Python package we need to install a Python interpreter. Considering that Python2 is soon end of life, we focus here on installing Python3.
The easiest way to install Python3 is with the Anaconda Python distribution.
Please download and install the Python3.x 64 bit distribution from `Anaconda <https://www.continuum.io/downloads>`_ and install it with the default settings.
After installation type in the Windows start bar ``anaconda`` and start the ``anaconda prompt``.


.. _install_opencl:

Installing OpenCL
-----------------
For OpenCL you need two things, an OpenCL driver for your platform and the Python OpenCL bindings.
Please make sure you install the correct OpenCL driver (Intel/AMD/NVidia) for your system. For graphics cards, please make sure you are using the
latest version of your graphics driver. For Intel processors download the drivers from https://software.intel.com/en-us/articles/opencl-drivers (only needed
if you want to run OpenCL on your CPU).

With the drivers installed and everything up to date, the Python PyOpenCL package needs to be installed next.
This package, named ``pyopencl``, can either be installed from a downloadable binary or using a Python software manager such as *pip*.
Using the binary is easiest since the installation with pip requires compilation which requires the Visual Studio 2015 (Community Edition or higher).

Binary OpenCL package
^^^^^^^^^^^^^^^^^^^^^
To install ``pyopencl`` using a binary file, please download the correct binary from http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl
(for example, download ``pyopencl-2016.2-cp35-cp35m-win_amd64.whl``). After the download, open an Anaconda Prompt (or a normal Windows command line) and
change directory to where you downloaded the ``.whl`` file and install the binary using pip:

.. code-block:: none

    > cd %UserProfile%\Downloads
    > pip install <filename>.whl

Please substitute ``<filename>`` for your downloaded filename.


Using pip and compilation
^^^^^^^^^^^^^^^^^^^^^^^^^
To install ``pyopencl`` using pip you must first install a version of Visual Studio 2015, with a few specific options enabled. First download the
Visual Studio 2015 Community edition and open the installer. When asked, choose for "Custom" installation and enable the following features:

* [] Programming Languages
    * [] Visual C++
        * [X] Common Tools for Visual C++ 2015
    * [X] Python Tools for Visual Studio

If you already have Visual Studio 2015 installed and are unsure if these options are enabled, you can rerun the installer and it will ask you if you
wish to update your installation with additional options.

With Visual Studio 2015 installed, installing pyopencl is easy. Open a Anaconda Prompt and type:

.. code-block:: none

    > pip install pyopencl


If all goes well, pyopencl will be compiled and installed to your system.


.. _install_mot:

Installing MOT
--------------
With Python and OpenCL installed you can now install MOT. Open an Anaconda Prompt and type:

.. code-block:: none

    > pip install MOT


Testing the installation
========================
Open a Python shell. In Windows you can do this using a the Anaconda Prompt and type ``python``. On Linux, use in Bash the ``python3`` command and in the prompt type:

.. code-block:: python

    >>> import mot
    >>> from mot.cl_environments import CLEnvironmentFactory
    >>> CLEnvironmentFactory.smart_device_selection()

If you get no errors and the output is a list of CL environments, MOT is succesfully installed.
