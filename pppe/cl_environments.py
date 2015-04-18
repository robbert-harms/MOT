import pyopencl as cl
from .tools import device_supports_double

__author__ = 'Robbert Harms'
__date__ = "2014-11-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLEnvironment(object):

    def __init__(self, platform, device, context=None, compile_flags=()):
        """Storage unit for an OpenCL environment.

        Args:
            platform (pyopencl platform): An PyOpenCL platform.
            device (pyopencl device): An PyOpenCL device
            context (pyopencl context): An PyOpenCL context
            compile_flags (list of str): A list of strings with compile flags (see the OpenCL specifications)

        Attributes:
            compile_flags (list of str): A list of strings with compile flags (see the OpenCL specifications)
        """
        self._platform = platform
        self._device = device
        self._context = context
        self.compile_flags = compile_flags

        if not self._context:
            self._context = cl.Context([self._device])

    def get_new_queue(self):
        """Create and return a new command queue

        Returns:
            CommandQueue: A command queue from PyOpenCL
        """
        return cl.CommandQueue(self._context, device=self._device)

    @property
    def supports_double(self):
        """Check if the device listed by this environment supports double

        Returns:
            boolean: True if the device supports double, false otherwise.
        """
        return device_supports_double(self.device)

    @property
    def platform(self):
        """Get the platform associated with this environment.

        Returns:
            pyopencl platform: The platform associated with this environment.
        """
        return self._platform

    @property
    def device(self):
        """Get the device associated with this environment.

        Returns:
            pyopencl device: The device associated with this environment.
        """
        return self._device

    @property
    def context(self):
        """Get the context associated with this environment.

        Returns:
            pyopencl context: The context associated with this environment.
        """
        return self._context

    @property
    def is_gpu(self):
        """Check if the device associated with this environment is a GPU.

        Returns:
            boolean: True if the device is an GPU, false otherwise.
        """
        return self._device.get_info(cl.device_info.TYPE) == cl.device_type.GPU

    @property
    def is_cpu(self):
        """Check if the device associated with this environment is a CPU.

        Returns:
            boolean: True if the device is an CPU, false otherwise.
        """
        return self._device.get_info(cl.device_info.TYPE) == cl.device_type.CPU

    def __repr__(self):
        s = "Platform: " + self._platform.name + "\n"
        s += "Vendor: " + self._platform.vendor + "\n"
        s += "Version: " + self._platform.version + "\n"
        s += "Compile flags: " + " ".join(self.compile_flags) + "\n"
        s += "Device: " + self._device.name + "\n"
        s += "\tIs GPU: " + repr(self.is_gpu) + "\n"
        s += "\tSupports double: " + repr(self.supports_double) + "\n"
        s += "\tVersion: " + self._device.opencl_c_version + "\n"
        s += "\tMax. Compute Units: " + repr(self._device.max_compute_units) + "\n"
        s += "\tLocal Memory Size: " + repr(self._device.local_mem_size/1024) + "KB" + "\n"
        s += "\tGlobal Memory Size: " + repr(self._device.global_mem_size/(1024*1024)) + "MB" + "\n"
        s += "\tMax Alloc Size: " + repr(self._device.max_mem_alloc_size/(1024*1024)) + "MB" + "\n"
        s += "\tMax Work-group Size: " + repr(self._device.max_work_group_size) + "\n"
        dim = self._device.max_work_item_sizes
        s += "\tMax Work-item Dims: (" + repr(dim[0]) + " " + " ".join(map(str, dim[1:])) + ")" + "\n"
        return s


class CLEnvironmentFactory(object):

    @staticmethod
    def single_device(cl_device_type=cl.device_type.GPU, platform=None, compile_flags=('-cl-strict-aliasing',),
                      fallback_to_any_device_type=False):
        """Get a list containing a single device environment, for a device of the given type on the given platform.

        This will only fetch devices that support double (possibly only double with a pragma
        defined, but still, it should support double).

        Args:
            cl_device_type (cl.device_type.* or string): The type of the device we want,
                can be a opencl device type or a string matching 'GPU' or 'CPU'.
            platform (opencl platform): The opencl platform to select the devices from
            compile_flags (list of str): A tuple with compile flags to use for this device / context.
            fallback_to_any_device_type (boolean): If True, try to fallback to any possible device in the system.

        Returns:
            list of CLEnvironment: List with one element, the CL runtime environment requested.
        """
        if cl_device_type == 'GPU':
            cl_device_type = cl.device_type.GPU
        elif cl_device_type == 'CPU':
            cl_device_type = cl.device_type.CPU

        if platform is None:
            platform = cl.get_platforms()[0]

        devices = platform.get_devices(device_type=cl_device_type)
        if not devices:
            if fallback_to_any_device_type:
                devices = platform.get_devices()
            else:
                raise ValueError('No devices of the specified type ({}) found.'.format(
                    cl.device_type.to_string(cl_device_type)))

        for dev in devices:
            if device_supports_double(dev):
                try:
                    env = CLEnvironment(platform, dev, compile_flags=compile_flags)
                    return [env]
                except cl.RuntimeError:
                    pass

        raise ValueError('No suitable OpenCL device found.')

    @staticmethod
    def all_devices(cl_device_type=None, platform=None, compile_flags=('-cl-strict-aliasing',)):
        """Get multiple device environments, optionally only of the indicated type.

        This will only fetch devices that support double (possibly only devices
        with a pragma defined, but still it should support double).

        Args:
            cl_device_type (cl.device_type.* or string): The type of the device we want,
                can be a opencl device type or a string matching 'GPU' or 'CPU'.
            platform (opencl platform): The opencl platform to select the devices from
            compile_flags (list of str): A tuple with compile flags to use for this device / context.

        Returns:
            list of CLEnvironment: List with one element, the CL runtime environment requested.
        """
        if cl_device_type is not None:
            if cl_device_type.upper() == 'GPU':
                cl_device_type = cl.device_type.GPU
            elif cl_device_type.upper() == 'CPU':
                cl_device_type = cl.device_type.CPU
            elif cl_device_type.upper() == 'ALL':
                cl_device_type = None

        runtime_list = []

        if platform is None:
            platforms = cl.get_platforms()
        else:
            platforms = [platform]

        for platform in platforms:
            if cl_device_type:
                devices = platform.get_devices(device_type=cl_device_type)
            else:
                devices = platform.get_devices()

            for device in devices:
                if device_supports_double(device):
                    try:
                        env = CLEnvironment(platform, device, compile_flags=compile_flags)
                        runtime_list.append(env)
                    except cl.RuntimeError:
                        pass

        return runtime_list