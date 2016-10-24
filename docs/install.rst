Installation
************


Ubuntu / Debian Linux
=====================
Using the package manager, installation in Ubuntu and Debian is easy.

For **Ubuntu >= 16** the MOT package can be installed with a Personal Package Archive (PPA) using:

.. code-block:: bash

    $ sudo add-apt-repository ppa:robbert-harms/cbclab
    $ sudo apt-get update
    $ sudo apt-get install python3-mot

Using such a PPA ensures that your Ubuntu system can update the MOT package automatically whenever a new version is out.
For **Debian**, and **Ubuntu < 16**, using a PPA is not possible and we need a more manual installation.
Please install the dependencies (*python3*, *pip3* and *pyopencl*) first:

.. code-block:: bash

    $ sudo apt-get install python3 python3-pip python3-pyopencl

and then install MOT with:

.. code-block:: bash

    $ sudo pip3 install MOT


After installation please continue with testing the installation below.


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
Installing ``pyopencl`` with pip requires Visual Studio 2015 and an OpenCL SDK to be present on your system.
First, install Visual Studio 2015 with a few specific options enabled (under "Custom" during the installation):

* [] Programming Languages
    * [] Visual C++
        * [X] Common Tools for Visual C++ 2015
    * [X] Python Tools for Visual Studio

If you already have Visual Studio 2015 installed and are unsure if these options are enabled, you can rerun the installer to update your installation with additional options.

After this installation please download and install an OpenCL software development kit (SDK) matching the vendor of your graphics card or processor:

* For Intel, see https://software.intel.com/en-us/intel-opencl
* For AMD, see http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/
* For NVidia, see https://developer.nvidia.com/cuda-downloads

With Visual Studio 2015 and an OpenCL SDK installed we can proceed to install PyOpenCL. Open an Anaconda Prompt or a Windows CMD and type:

.. code-block:: none

    > pip install pyopencl


If this completes without errors, PyOpenCL is installed. If you get compilation errors, please set the following environment variables according to your system and try again:

.. code-block:: none

    > set INCLUDE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include
    > set LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64
    > pip install pyopencl

The paths listed here assume an NVidia system. Please adapt the paths to your own system. The ``INCLUDE`` path should contain the file ``CL\cl.h`` and the ``LIB`` path
should contain ``OpenCL.lib``. If all goes well, pyopencl will be compiled and installed to your system.


.. _install_mot:

Installing MOT
--------------
With Python and OpenCL installed you can now install MOT. Open an Anaconda Prompt and type:

.. code-block:: none

    > pip install mot


Testing the installation
========================
Open a Python shell. In Windows you can do this using a the Anaconda Prompt and type ``python``. On Linux, use in Bash the ``python3`` command and in the prompt type:

.. code-block:: python

    >>> import mot
    >>> devices = mot.smart_device_selection()
    >>> list(map(str, devices))

If you get no errors and the output is a list of CL environments, MOT is successfully installed.



Upgrading
=========

Ubuntu / Debian Linux
---------------------
If you used the PPA to install the MOT package, upgrading is easy and is handled automatically by Ubuntu.
If you used the pip3 installation procedure you can upgrade MOT with ``sudo pip3 install --upgrade MOT``.


Windows
-------
To upgrade MOT when a new version is out, open an Anaconda Prompt or Windows CMD and type:

.. code-block:: none

    > pip install --upgrade mot

to upgrade MOT to the latest version.
