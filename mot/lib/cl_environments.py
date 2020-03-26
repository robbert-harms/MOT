import pyopencl as cl
from mot.lib.utils import device_supports_double, device_type_from_string

__author__ = 'Robbert Harms'
__date__ = "2014-11-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert@xkls.nl"


class CLEnvironment:

    def __init__(self, platform, context, device):
        """Storage unit for an OpenCL environment.

        Args:
            platform (cl.Platform): An PyOpenCL platform.
            context (cl.Context): The CL context
            device (cl.Device): The CL device
        """
        self._platform = platform
        self._context = context
        self._device = device
        self._queue = cl.CommandQueue(self._context, device=device)

    @property
    def context(self):
        """Get a CL context containing this device.

        Returns:
            cl.Context: a PyOpenCL device context
        """
        return self._context

    @property
    def queue(self):
        """Get a CL queue for this device and context.

        Returns:
            cl.Queue: a PyOpenCL queue
        """
        return self._queue

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

    @property
    def device_type(self):
        """Get the device type of the device in this environment.

        Returns:
            the device type of this device.
        """
        return self._device.get_info(cl.device_info.TYPE)

    def __str__(self):
        s = 'GPU' if self.is_gpu else 'CPU'
        s += ' - ' + self.device.name + ' (' + self.platform.name + ')'
        return s

    def __repr__(self):
        s = 75*"=" + "\n"
        s += repr(self._platform) + "\n"
        s += 75*"=" + "\n"
        s += self._print_info(self._platform, cl.platform_info)

        s += 75*"-" + "\n"
        s += repr(self._device) + "\n"
        s += 75*"-" + "\n"
        s += self._print_info(self._device, cl.device_info)

        return s

    def _print_info(self, obj, info_cls):
        s = ''

        def format_title(title_str):
            title_str = title_str.lower()
            title_str = title_str.replace('_', ' ')
            return title_str

        for info_name in sorted(dir(info_cls)):
            if not info_name.startswith("_") and info_name != "to_string":
                info = getattr(info_cls, info_name)

                try:
                    info_value = obj.get_info(info)
                except cl.LogicError:
                    info_value = "<error>"

                if info_cls == cl.device_info and info_name == "PARTITION_TYPES_EXT" and isinstance(info_value, list):
                    prop_value = [cl.device_partition_property_ext.to_string(v, "<unknown device "
                                                                                "partition property %d>")
                                  for v in info_value]

                    s += ("%s: %s" % (format_title(info_name), prop_value)) + "\n"
                else:
                    try:
                        s += ("%s: %s" % (format_title(info_name), info_value)) + "\n"
                    except cl.LogicError:
                        s += ("%s: <error>" % info_name) + "\n"
        s += "\n"
        return s

    def __hash__(self):
        return hash(self._platform) + hash(self._context) + hash(self._device) + hash(self._queue)


def _initialize_cl_environment_cache():
    """Initialize a cache of CL environments.

    This cache holds, per platform a cache with all applicable CL environments.

    Returns:
        dict: a dictionary mapping platforms to CLEnvironment
    """
    cache = {}
    for platform in cl.get_platforms():
        devices = platform.get_devices()

        if len(devices) == 0:
            continue

        try:
            context = cl.Context(devices)
        except cl.RuntimeError:
            continue
        items = []
        for device in devices:
            if device_supports_double(device):
                env = CLEnvironment(platform, context, device)
                items.append(env)
        cache[platform] = items

    return cache


_cl_environment_cache = _initialize_cl_environment_cache()


class CLEnvironmentFactory:

    @staticmethod
    def single_device(cl_device_type='GPU', platform=None, fallback_to_any_device_type=False):
        """Get a list containing a single device environment, for a device of the given type on the given platform.

        This will only fetch devices that support double (possibly only double with a pragma
        defined, but still, it should support double).

        Args:
            cl_device_type (cl.device_type.* or string): The type of the device we want,
                can be a opencl device type or a string matching 'GPU', 'CPU' or 'ALL'.
            platform (opencl platform): The opencl platform to select the devices from
            fallback_to_any_device_type (boolean): If True, try to fallback to any possible device in the system.

        Returns:
            list of CLEnvironment: List with one element, the CL runtime environment requested.
        """
        cl_environments = CLEnvironmentFactory.all_devices(cl_device_type=cl_device_type, platform=platform)

        if len(cl_environments):
            return cl_environments[0]
        else:
            if fallback_to_any_device_type:
                return CLEnvironmentFactory.single_device(cl_device_type=None)
            else:
                raise ValueError('No suitable devices of the specified type ({}) found.'.format(
                    cl.device_type.to_string(cl_device_type)))

    @staticmethod
    def all_devices(cl_device_type=None, platform=None):
        """Get multiple device environments, optionally only of the indicated type.

        This will only fetch devices that support double point precision.

        Args:
            cl_device_type (cl.device_type.* or string): The type of the device we want,
                can be a opencl device type or a string matching 'GPU' or 'CPU'.
            platform (opencl platform): The opencl platform to select the devices from

        Returns:
            list of CLEnvironment: List with the CL device environments.
        """
        if isinstance(cl_device_type, str):
            cl_device_type = device_type_from_string(cl_device_type)

        cl_environments = []

        if platform is None:
            platforms = _cl_environment_cache.keys()
        else:
            platforms = [platform]

        for platform in platforms:
            cached_envs = _cl_environment_cache[platform]

            if cl_device_type:
                for env in cached_envs:
                    if env.device.get_info(cl.device_info.TYPE) == cl_device_type:
                        cl_environments.append(env)
            else:
                cl_environments.extend(cached_envs)

        return cl_environments

    @staticmethod
    def smart_device_selection(preferred_device_type=None):
        """Get a list of device environments that is suitable for use in MOT.

        Basically this gets the total list of devices using all_devices() and applies a filter on it.

        This filter does the following:
            1) if the 'AMD Accelerated Parallel Processing' is available remove all environments using the 'Clover'
                platform.

        More things may be implemented in the future.

        Args:
            preferred_device_type (str): the preferred device type, one of 'CPU', 'GPU' or 'APU'.
                If no devices of this type can be found, we will use any other device available.

        Returns:
            list of CLEnvironment: List with the CL device environments.
        """
        cl_environments = CLEnvironmentFactory.all_devices(cl_device_type=preferred_device_type)
        platform_names = [env.platform.name for env in cl_environments]
        has_amd_pro_platform = any('AMD Accelerated Parallel Processing' in name for name in platform_names)

        if has_amd_pro_platform:
            return list(filter(lambda env: 'Clover' not in env.platform.name, cl_environments))

        if preferred_device_type is not None and not len(cl_environments):
            return CLEnvironmentFactory.all_devices()

        return cl_environments
