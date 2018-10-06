############
Installation
############


*********************
Ubuntu / Debian Linux
*********************
Using the package manager, installation in Ubuntu and Debian is relatively straightforward.

For **Ubuntu >= 16** the MOT package can be installed with a Personal Package Archive (PPA) using:

.. code-block:: bash

    $ sudo add-apt-repository ppa:robbert-harms/cbclab
    $ sudo apt update
    $ sudo apt install python3-mot

Using such a PPA ensures that your Ubuntu system can update the MOT package automatically whenever a new version is out.
For **Debian**, and **Ubuntu < 16**, using a PPA is not possible and we need a more manual installation.
Please install the dependencies (*python3*, *pip3* and *pyopencl*) first:

.. code-block:: bash

    $ sudo apt install python3 python3-pip python3-pyopencl python3-devel

and then install MOT with:

.. code-block:: bash

    $ sudo pip3 install mot


After installation please continue with testing the installation below.


***
Mac
***
Installation on Mac is pretty easy using the Anaconda 4.2 or higher Python distribution.
Please download and install the Python3.x 64 bit distribution, version 4.2 or higher which includes PyQt5,
from `Anaconda <https://www.continuum.io/downloads>`_ and install it with the default settings.

Afterwards, open a terminal and type:

.. code-block:: bash

    $ pip install mot


To install MOT to your system.


*******
Windows
*******
The installation on Windows is a little bit more convoluted due to the lack of a package manager. The installation is a multi-step procedure:

1. Installing a :ref:`Python interpreter <install_python>`
2. Installing the :ref:`OpenCL drivers <install_opencl>`
3. Installing the :ref:`Python OpenCL bindings PyOpenCL <install_pyopencl>`
4. :ref:`Installing MOT <install_mot>`


.. _install_python:

Installing Python
=================
Since MOT is a Python package we need to install a Python interpreter.
Considering that Python2 is soon end-of-life, this package only supports Python3.

The easiest way to install Python3 is with the Anaconda Python distribution.
Please download and install the Python3.x 64 bit distribution, version 4.2 or higher which includes PyQt5, from `Anaconda <https://www.continuum.io/downloads>`_ and install it with the default settings.
If you are following this guide with the intention of installing `MDT <https://github.com/cbclab/MDT>`_ afterwards, please note that Anaconda versions prior to 4.2 have the (deprecated) PyQt4 as its Qt library.
This is not a problem for MOT per se.
However if you want to install MDT and use its Qt5 GUI, or more generally want to use Qt5 and packages that depend on Qt5, you will find benefit from installing Anaconda > 4.2 with PyQt5.
If you insist on using an older Anaconda install or PyQt4 `environment <https://conda.io/docs/using/envs.html>`_ (also consider creating a new PyQt5 compatible env), note that this is possible, but you will have to install a PyQt5 package yourself, such as the m-labs PyQt5 Anaconda package and deal with its version conflicts, e.g. python version <= 3.4.


After installation type ``Anaconda Prompt`` in the Windows start bar and start the Anaconda Prompt command line interface.


.. _install_opencl:

Installing OpenCL drivers
=========================
To run OpenCL applications you need an OpenCL driver for your platform.
Please download and install the correct device driver (Intel/AMD/NVidia) for your device with support for OpenCL 1.2 or higher.
For graphics cards, make sure you are using the latest version of your graphics driver.
For Intel processors download the OpenCL runtime from https://software.intel.com/en-us/articles/opencl-drivers
(OpenCL Runtime for Intel Core and Intel Xeon Processors; towards the end).
Note that installing the Intel driver is needed if you want to run OpenCL on your Intel CPUs. Is is not needed if you only want to run on your GPUs.
As a rule, you need to have an OpenCL driver or runtime installed for every device you want to run computations on.
Most often, having both CPU and GPU available is desirable.

.. _install_pyopencl:

Installing PyOpenCL
===================
With the drivers installed and everything up to date, we can now proceed with installing the Python OpenCL bindings, ``pyopencl``.
This is often the most problematic step and errors later on (e.g. in testing MOT) often come down to an incomplete (failed)
or incompatible (successful but not working) pyopencl package install.
PyOpenCL can either be installed from a downloadable binary or be compiled from source.
Using the binary is easiest since manual compilation is more difficult.


Alternative 1: Using a binary PyOpenCL package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Installing a precompiled binary wheel (.whl) is the easiest way to install PyOpenCL, but only works if the wheel is compiled for your specific Python implementation.
At Christoph Gohlke website (http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl) you can find a range of PyOpenCL binary packages.
If there is a compatible one for your system, download that version.
You can see if it is compatible if the Python version and OpenCl version in the binary name matches that of your installed Python and driver supported versions. 
Note that many drivers, such as nVIDIA’s only support OpenCL 1.2, so in that case take the wheel with "+cl12" in the name, and not e.g. "+cl21". For example if you have
64-bit Windows system with Python 3.5 and your GPU or CPU drivers support OpenCL 1.2 you need to download the wheel with "+cl12", ``win-amd64`` and cp35m in the name
(note the format, cp<version>m, the m is important).
(To check which Python version you have you can run ``python --version`` in the command line).

If there is no compatible version for your system to be found on Gohlke's website, here is a mirror of an older version by Gohlke that is compatible with most Python 3.5 systems:
:download:`pyopencl-2015.2.4-cp35-none-win_amd64 <./_downloads/pyopencl-2015.2.4-cp35-none-win_amd64.whl>`.

After the download, open an Anaconda Prompt (or a normal Windows cmd) as administrator (right-click the command and select "Run as administrator") and change directory to where you downloaded the ``.whl`` file.
Then, install the binary using pip:

.. code-block:: none

    > cd %UserProfile%\Downloads
    > pip install <filename>.whl

Please make sure you are in the right directory and please substitute ``<filename>`` for your downloadeded filename.

To test if this binary package works, open a Python shell (for instance by typing ``python`` in your open prompt) and type:

.. code-block:: python

    >>> import pyopencl

If that works (python ``>>>`` prompt reappears without messages about missing dll's and cffi problems), you are good to go. (exit the python prompt by typing ``exit()`` or Ctrl-Z then Enter)
If you encounter an error that ends on something like:

.. code-block:: none

    > ImportError: DLL load failed: The specified procedure could not be found.

Then the binary package (.whl file) is not compatible with your OS version and/or Python installation.
Either try a different wheel, or try the compilation procedure below.


Alternative 2: Compile PyOpenCL with Visual Studio 15
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Installing ``pyopencl`` with pip from source code requires Visual Studio 2015 and an OpenCL SDK (this is different from a driver or runtime, the SDK includes compilation header files) to be present on your system.
First, install Visual Studio 2015 with a few specific options enabled (under "Custom" during the installation):

* [] Programming Languages
    * [] Visual C++
        * [X] Common Tools for Visual C++ 2015
    * [X] Python Tools for Visual Studio

If you already have Visual Studio 2015 installed and are unsure if these options are enabled, you can rerun the installer to update your installation with additional options.

After this installation please download and install an OpenCL software development kit (SDK) matching the vendor of your graphics card or processor:

* For Intel, see https://software.intel.com/en-us/intel-opencl
* For AMD, see https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases
* For NVidia, see https://developer.nvidia.com/cuda-downloads

With Visual Studio 2015 and an OpenCL SDK installed we can proceed to install PyOpenCL. Open an Anaconda Prompt or a Windows CMD and type:

.. code-block:: none

    > pip install pyopencl


If this completes without errors, PyOpenCL is installed. If you get compilation errors, please set the INCLUDE and LIB environment variables according to your system and try again, e.g. for the CUDA 8 SDK use:

.. code-block:: none

    > set INCLUDE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include
    > set LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64
    > pip install pyopencl

The paths listed here assume an NVidia system. Please adapt the paths to your own system and device SDK (e.g. ATI). Important is that the ``INCLUDE`` path should contain
the file ``CL\cl.h`` and the ``LIB`` path should contain ``OpenCL.lib``. Find these directories if needed. If all goes well, PyOpenCL will be compiled and installed to your system.

If this still does not work, you can try one of the installation guides on https://wiki.tiker.net/PyOpenCL/Installation/Windows, or you can consider (re)installing Anaconda, version >=4.2, with Python 3.5 on your 64-bit Windows system and then try the -cp35-none-win_amd64 wheel linked above.


.. _install_mot:

Installing MOT
==============
With Python and OpenCL installed you can now install MOT. Open an Anaconda Prompt and type:

.. code-block:: none

    > pip install mot

************************
Testing the installation
************************
Open a Python shell. In Windows you can do this using a the Anaconda Prompt and type ``python``. On Linux, use in Bash the ``python3`` command. In the prompt type:

.. code-block:: python

    >>> import mot
    >>> devices = mot.smart_device_selection()
    >>> list(map(str, devices))

If you get no errors and the output is a list of CL environments, MOT is successfully installed.


*********
Upgrading
*********

Ubuntu / Debian Linux
=====================
If you used the PPA to install the MOT package, upgrading is easy and is handled automatically by Ubuntu.
If you used the pip3 installation procedure you can upgrade MOT with ``sudo pip3 install --upgrade MOT``.


Windows
=======
To upgrade MOT when a new version is out, open an Anaconda Prompt or Windows CMD and type:

.. code-block:: none

    > pip uninstall mot
    > pip install mot

to upgrade MOT to the latest version.
