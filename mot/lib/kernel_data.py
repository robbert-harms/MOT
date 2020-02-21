import numbers
from collections import OrderedDict, Mapping

import numpy as np
import pyopencl as cl

from mot.lib.cl_environments import CLEnvironment
from mot.lib.utils import dtype_to_ctype, ctype_to_dtype, convert_data_to_dtype, is_vector_ctype, split_vector_ctype

__author__ = 'Robbert Harms'
__date__ = '2018-04-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'


class KernelData:

    @property
    def ctype(self):
        """Get the ctype of this kernel data element

        Returns:
            str: the ctype of this kernel data element
        """
        raise NotImplementedError()

    def get_subset(self, problem_indices=None, batch_range=None):
        """Get a subset of this kernel data over problem instances.

        This can be used to get a subset of a kernel data object such that only a subset of the problem instances
        are processed.

        One can either provide a list of indices or a range defined by a start and end position. Providing a list
        of indices is more flexible, but will incur a memory copy. Providing a range only allows contiguous slices,
        but it will use the original memory.

        Args:
            problem_indices (Iterable or None): list of problem instances we would like to have in the subset.
                May return ``self``. If problem_indices evaluates to None, this should return ``self``.
            batch_range (Tuple(int, int)): the range specified by a starting and ending position. The
                Start position is taken inclusive, the end position is taken exclusive.

        Returns:
            KernelData: a kernel data object of the same type as the current kernel data object.
        """
        raise NotImplementedError()

    def set_mot_float_dtype(self, mot_float_dtype):
        """Set the numpy data type corresponding to the ``mot_float_type`` ctype.

        This is set just prior to using this kernel data in the kernel.

        Args:
            mot_float_dtype (dtype): the numpy data type that is to correspond with the ``mot_float_type`` used in the
                kernels.
        """
        raise NotImplementedError()

    def get_data(self):
        """Get the underlying data of this kernel data object.

        Returns:
            dict, ndarray, scalar: the underlying data object, can return None if this input data has no actual data.
        """
        raise NotImplementedError()

    def get_children(self):
        """Get a list of children kernel data elements.

        Returns:
            List[KernelData]: list of children kernel data elements, can be an an empty list.
        """
        raise NotImplementedError()

    def get_scalar_arg_dtypes(self):
        """Get the numpy data types we should report in the kernel call for scalar elements.

        If we are inserting scalars in the kernel we need to provide the CL runtime with the correct data type
        of the function. If the kernel parameter is not a scalar, we should return None. If the kernel data does not
        require a kernel input parameter, return an empty list.

        This list should match the list of parameters of :meth:`get_kernel_parameters`.

        Returns:
            List[Union[dtype, None]]: the numpy data type for this element, or None if this is not a scalar.
        """
        raise NotImplementedError()

    def get_type_definitions(self):
        """Get possible type definitions needed to load this data into the kernel.

        These types are defined at the head of the CL script, before any functions.

        Returns:
            str: a CL compatible type declaration. This can for example be used for defining struct types.
                If no extra types are needed, this function should return the empty string.
        """
        raise NotImplementedError()

    def initialize_variable(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        """Initialize the variable inside the kernel function.

        This should initialize the variable as such that we can use it when calling the function acting on this data.

        Args:
             variable_name (str): the name for this variable
             kernel_param_name (str): the kernel parameter name (given in :meth:`get_kernel_parameters`).
             problem_id_substitute (str): the substitute for the ``{problem_id}`` in the kernel data info elements.
             address_space (str): the desired address space for this variable, defined by the parameter of the called
                function.

        Returns:
            str: the necessary CL code to initialize this variable
        """
        raise NotImplementedError()

    def get_function_call_input(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        """How this kernel data is used as input to the function that operates on the data.

        Args:
            variable_name (str): the name for this variable
            kernel_param_name (str): the kernel parameter name (given in :meth:`get_kernel_parameters`).
            problem_id_substitute (str): the substitute for the ``{problem_id}`` in the kernel data info elements.
            address_space (str): the desired address space for this variable, defined by the parameter of the called
                function.

        Returns:
            str: a single string representing how this kernel data is used as input to the function we are applying
        """
        raise NotImplementedError()

    def post_function_callback(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        """A callback to update or change data after the function has been applied

        Args:
            variable_name (str): the name for this variable
            kernel_param_name (str): the kernel parameter name (given in :meth:`get_kernel_parameters`).
            problem_id_substitute (str): the substitute for the ``{problem_id}`` in the kernel data info elements.
            address_space (str): the desired address space for this variable, defined by the parameter of the called
                function.

        Returns:
            str: CL code that needs to be run after the function has been applied.
        """
        raise NotImplementedError()

    def get_struct_declaration(self, name):
        """Get the variable declaration of this data object for use in a Struct.

        Args:
            name (str): the name for this data

        Returns:
            str: the variable declaration of this kernel data object
        """
        raise NotImplementedError()

    def get_struct_initialization(self, variable_name, kernel_param_name, problem_id_substitute):
        """Initialize the variable inside a struct.

        This should initialize the variable for use in a struct (should correspond to :meth:`get_struct_declaration`).

        Args:
             variable_name (str): the name for this variable
             kernel_param_name (str): the kernel parameter name (given in :meth:`get_kernel_parameters`).
             problem_id_substitute (str): the substitute for the ``{problem_id}`` in the kernel data info elements.

        Returns:
            str: the necessary CL code to initialize this variable
        """
        raise NotImplementedError()

    def get_kernel_parameters(self, kernel_param_name):
        """Get the kernel argument declarations for this kernel data.

        Args:
            kernel_param_name (str): the parameter name for the parameter in the kernel function call

        Returns:
            List[str]: a list of kernel parameter declarations, or an empty list
        """
        raise NotImplementedError()

    def enqueue_host_access(self, cl_environments, is_blocking=True, wait_for=None):
        """Enqueue either a map or write operation for this kernel input data object.

        This should add non-blocking maps or write operations to the given queue.

        Args:
            cl_environments (mot.lib.cl_environments.CLEnvironment or list):
                enqueue host access for one or multiple CL environments
            is_blocking (boolean): if the enqueuing should be blocking calls or not
                This first enqueues all the work to all the contexts and then waits for completion on them.
            wait_for (Dict[CLEnvironment: cl.Event]): per CL environment an event to wait on

        Returns:
            Dict[CLEnvironment: cl.Event]: per CLEnvironment an event for waiting on
        """
        raise NotImplementedError()

    def enqueue_device_access(self, cl_environments, is_blocking=True, wait_for=None):
        """Enqueue either an unmap or read operation for this kernel input data object.

        This should add non-blocking unmaps or read operations to the given queue.

        Args:
            cl_environments (mot.lib.cl_environments.CLEnvironment or list):
                enqueue device access for one or multiple CL environments
            is_blocking (boolean): if the enqueuing should be blocking calls or not
                This first enqueues all the work to all the contexts and then waits for completion on them.
            wait_for (Dict[CLEnvironment: cl.Event]): per CL environment an event to wait on

        Returns:
            Dict[CLEnvironment: cl.Event]: per CLEnvironment an event for waiting on
        """
        raise NotImplementedError()

    def get_kernel_inputs(self, cl_environment, workgroup_size):
        """Get the kernel input data matching the list of parameters of :meth:`get_kernel_parameters`.

        Since the kernels follow the map/unmap paradigm make sure to use the ``USE_HOST_PTR`` when making
        writable data objects.

        Args:
            cl_environment (mot.lib.cl_environments.CLEnvironment): the environment for which to prepare the input
            workgroup_size (int): the workgroup size the kernel will use.

        Returns:
            List: a list of buffers, local memory objects, scalars, etc., anything that can be loaded into the kernel.
                If no data should be entered, return an empty list.
        """
        raise NotImplementedError()

    def get_nmr_kernel_inputs(self):
        """Get the number of kernel inputs this input data object has.

        Returns:
            int: the number of kernel inputs
        """
        raise NotImplementedError()


class Struct(KernelData):

    def __init__(self, elements, ctype, anonymous=False):
        """A kernel data element for structs.

        Please be aware that structs will always be passed as a pointer to the calling function.

        Args:
            elements (Dict[str, Union[Dict, KernelData]]): the kernel data elements to load into the kernel
                Can be a nested dictionary, in which case we load the nested elements as anonymous structs.
                Alternatively, you can nest Structs in Structs, yielding named structs.
            ctype (str): the name of this structure
            anonymous (boolean): if this struct is to be loaded anonymously, this is only meant for nested Structs.
        """
        self._elements = OrderedDict(sorted(elements.items()))

        for key in list(self._elements):
            value = self._elements[key]
            if isinstance(value, Mapping):
                self._elements[key] = Struct(value, key, anonymous=True)

        self._ctype = ctype
        self._anonymous = anonymous

    @property
    def ctype(self):
        return self._ctype

    def get_subset(self, problem_indices=None, batch_range=None):
        if problem_indices is None and batch_range is None:
            return self
        sub_elements = OrderedDict([(k, v.get_subset(problem_indices, batch_range))
                                    for k, v in self._elements.items()])
        return Struct(sub_elements, self._ctype, anonymous=self._anonymous)

    def set_mot_float_dtype(self, mot_float_dtype):
        for element in self._elements.values():
            element.set_mot_float_dtype(mot_float_dtype)

    def get_data(self):
        data = {}
        for name, value in self._elements.items():
            data[name] = value.get_data()
        return data

    def get_children(self):
        return self._elements.values()

    def get_scalar_arg_dtypes(self):
        dtypes = []
        for d in self._elements.values():
            dtypes.extend(d.get_scalar_arg_dtypes())
        return dtypes

    def get_type_definitions(self):
        other_structs = '\n'.join(element.get_type_definitions() for element in self._elements.values())

        if self._anonymous:
            return other_structs

        return other_structs + '''
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            typedef struct {ctype}{{
                {definitions}
            }} {ctype};
            #endif // {inclusion_guard_name}
        '''.format(ctype=self._ctype,
                   definitions='\n'.join(data.get_struct_declaration(name) for name, data in self._elements.items()),
                   inclusion_guard_name='INCLUDE_GUARD_CL_EXTRA_{}'.format(self._ctype))

    def initialize_variable(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return_str = ''
        for name, data in self._elements.items():
            return_str += data.initialize_variable('{}_{}'.format(variable_name, name),
                                                   '{}_{}'.format(kernel_param_name, name),
                                                   problem_id_substitute, 'global')

        inits = [data.get_struct_initialization(
            '{}_{}'.format(variable_name, name),
            '{}_{}'.format(kernel_param_name, name), problem_id_substitute) for name, data in self._elements.items()]

        if self._anonymous:
            return return_str + '''
                struct {ctype}{{
                    {definitions}
                }};
                struct {ctype} {v_name} = {{ {inits} }};
            '''.format(ctype=self._ctype, v_name=variable_name, inits=', '.join(inits),
                       definitions='\n'.join(data.get_struct_declaration(name)
                                             for name, data in self._elements.items()))

        return return_str + '''
            {ctype} {v_name} = {{ {inits} }};
        '''.format(ctype=self._ctype, v_name=variable_name, inits=', '.join(inits))

    def get_function_call_input(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return '&' + variable_name

    def post_function_callback(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return ''

    def get_struct_declaration(self, name):
        if self._anonymous:
            return '''
                struct {{
                    {definitions}
                }}* {name};
            '''.format(
                name=name,
                definitions='\n'.join(data.get_struct_declaration(name) for name, data in self._elements.items()))
        return '{}* {};'.format(self._ctype, name)

    def get_struct_initialization(self, variable_name, kernel_param_name, problem_id_substitute):
        if self._anonymous:
            return '(void*)(&{})'.format(variable_name)
        return '&' + variable_name

    def get_kernel_parameters(self, kernel_param_name):
        parameters = []
        for name, d in self._elements.items():
            parameters.extend(d.get_kernel_parameters('{}_{}'.format(kernel_param_name, name)))
        return parameters

    def enqueue_host_access(self, cl_environments, is_blocking=True, wait_for=None):
        events = {}
        for d in self._elements.values():
            events.update(d.enqueue_host_access(cl_environments, is_blocking=is_blocking, wait_for=wait_for))
        return events

    def enqueue_device_access(self, cl_environments, is_blocking=True, wait_for=None):
        events = {}
        for d in self._elements.values():
            events.update(d.enqueue_device_access(cl_environments, is_blocking=is_blocking, wait_for=wait_for))
        return events

    def get_kernel_inputs(self, cl_environment, workgroup_size):
        data = []
        for d in self._elements.values():
            data.extend(d.get_kernel_inputs(cl_environment, workgroup_size))
        return data

    def get_nmr_kernel_inputs(self):
        return sum(element.get_nmr_kernel_inputs() for element in self._elements.values())

    def __getitem__(self, key):
        return self._elements[key]

    def __contains__(self, key):
        return key in self._elements

    def __len__(self):
        return len(self._elements)


class Scalar(KernelData):

    def __init__(self, value, ctype=None, inline=True):
        """A kernel input scalar.

        This will insert the given value directly into the kernel's source code, and will not load it as a buffer.

        Args:
            value (number): the number to insert into the kernel as a scalar.
            ctype (str): the desired c-type for in use in the kernel, like ``int``, ``float`` or ``mot_float_type``.
                If None it is implied from the value.
            inline (bool): if set to True, we inline the value in the generated kernel code,
                else we set it as an argument.
        """
        if isinstance(value, str) and value == 'INFINITY':
            self._value = np.inf
        elif isinstance(value, str) and value == '-INFINITY':
            self._value = -np.inf
        else:
            self._value = np.array(value)
        self._ctype = ctype or dtype_to_ctype(self._value.dtype)
        self._mot_float_dtype = None
        self._inline = inline

    @property
    def ctype(self):
        return self._ctype

    def get_subset(self, problem_indices=None, batch_range=None):
        return self

    def get_data(self):
        if self._ctype.startswith('mot_float_type'):
            return np.asscalar(self._value.astype(self._mot_float_dtype))
        return np.asscalar(self._value)

    def get_children(self):
        return []

    def get_scalar_arg_dtypes(self):
        if self._inline:
            return []
        return [ctype_to_dtype(self._ctype)]

    def get_type_definitions(self):
        return ''

    def get_struct_declaration(self, name):
        return '{} {};'.format(self._ctype, name)

    def get_struct_initialization(self, variable_name, kernel_param_name, problem_id_substitute):
        if self._inline:
            return self.get_function_call_input(variable_name, kernel_param_name, problem_id_substitute, 'private')
        return kernel_param_name

    def get_kernel_parameters(self, kernel_param_name):
        if self._inline:
            return []
        return ['{} {}'.format(self._ctype, kernel_param_name)]

    def get_kernel_inputs(self, cl_environment, workgroup_size):
        if self._inline:
            return []
        return [self.get_data()]

    def get_nmr_kernel_inputs(self):
        if self._inline:
            return 0
        return 1

    def set_mot_float_dtype(self, mot_float_dtype):
        self._mot_float_dtype = mot_float_dtype

    def initialize_variable(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return ''

    def enqueue_host_access(self, cl_environments, is_blocking=True, wait_for=None):
        return {}

    def enqueue_device_access(self, cl_environments, is_blocking=True, wait_for=None):
        return {}

    def get_function_call_input(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        if not self._inline:
            return kernel_param_name

        if is_vector_ctype(self._ctype):
            vector_length = split_vector_ctype(self._ctype)[1]

            values = [str(el) for el in np.atleast_1d(np.squeeze(self._value))]

            if len(values) < vector_length:
                values.extend(['0'] * (vector_length - len(values)))
            assignment = '(' + self._ctype + ')(' + ', '.join(values) + ')'

        elif np.isposinf(self._value):
            assignment = 'INFINITY'
        elif np.isneginf(self._value):
            assignment = '-INFINITY'
        else:
            assignment = str(np.squeeze(self._value))
        return assignment

    def post_function_callback(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return ''


class PrivateMemory(KernelData):

    def __init__(self, nmr_items, ctype):
        """Adds a private memory array of the indicated size to the kernel data elements.

        This is useful if you want to have private memory arrays in kernel data structs.

        Args:
            nmr_items (int): the size of the private memory array
            ctype (str): the desired c-type for this local memory object, like ``int``, ``float`` or ``mot_float_type``.
        """
        self._ctype = ctype
        self._mot_float_dtype = None
        self._nmr_items = nmr_items

    @property
    def ctype(self):
        return self._ctype

    def get_subset(self, problem_indices=None, batch_range=None):
        return self

    def set_mot_float_dtype(self, mot_float_dtype):
        self._mot_float_dtype = mot_float_dtype

    def get_data(self):
        return None

    def get_children(self):
        return []

    def get_scalar_arg_dtypes(self):
        return []

    def get_type_definitions(self):
        return ''

    def initialize_variable(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return '''
            {ctype} {v_name}[{nmr_elements}];
        '''.format(ctype=self._ctype, v_name=kernel_param_name, nmr_elements=self._nmr_items)

    def get_function_call_input(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return kernel_param_name

    def post_function_callback(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return ''

    def get_struct_declaration(self, name):
        return '{}* {};'.format(self._ctype, name)

    def get_struct_initialization(self, variable_name, kernel_param_name, problem_id_substitute):
        return kernel_param_name

    def get_kernel_parameters(self, kernel_param_name):
        return []

    def enqueue_host_access(self, cl_environments, is_blocking=True, wait_for=None):
        return {}

    def enqueue_device_access(self, cl_environments, is_blocking=True, wait_for=None):
        return {}

    def get_kernel_inputs(self, cl_environment, workgroup_size):
        return []

    def get_nmr_kernel_inputs(self):
        return 0


class LocalMemory(KernelData):

    def __init__(self, ctype, nmr_items=None):
        """Indicates that a local memory array of the indicated size must be loaded as kernel input data.

        By default, this will create a local memory object the size of the local work group.

        Args:
            ctype (str): the desired c-type for this local memory object, like ``int``, ``float`` or ``mot_float_type``.
            nmr_items (int or Callable[[int], int]): either the size directly or a function that can calculate the
                required local memory size given the work group size. This will independently be multiplied with the
                item size of the ctype for the final size in bytes.
        """
        self._ctype = ctype
        self._mot_float_dtype = None

        if nmr_items is None:
            self._size_func = lambda workgroup_size: workgroup_size
        elif isinstance(nmr_items, numbers.Number):
            self._size_func = lambda _: nmr_items
        else:
            self._size_func = nmr_items

    @property
    def ctype(self):
        return self._ctype

    def get_subset(self, problem_indices=None, batch_range=None):
        return self

    def set_mot_float_dtype(self, mot_float_dtype):
        self._mot_float_dtype = mot_float_dtype

    def get_data(self):
        return None

    def get_children(self):
        return []

    def get_scalar_arg_dtypes(self):
        return [None]

    def get_type_definitions(self):
        return ''

    def initialize_variable(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return ''

    def get_function_call_input(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return kernel_param_name

    def post_function_callback(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return ''

    def get_struct_declaration(self, name):
        return 'local {}* restrict {};'.format(self._ctype, name)

    def get_struct_initialization(self, variable_name, kernel_param_name, problem_id_substitute):
        return self.get_function_call_input(variable_name, kernel_param_name, problem_id_substitute, '')

    def get_kernel_parameters(self, kernel_param_name):
        return ['local {}* restrict {}'.format(self._ctype, kernel_param_name)]

    def enqueue_host_access(self, cl_environments, is_blocking=True, wait_for=None):
        return {}

    def enqueue_device_access(self, cl_environments, is_blocking=True, wait_for=None):
        return {}

    def get_kernel_inputs(self, cl_environment, workgroup_size):
        mot_float_type_dtype = None
        if self._mot_float_dtype:
            mot_float_type_dtype = dtype_to_ctype(self._mot_float_dtype)

        itemsize = np.dtype(ctype_to_dtype(self._ctype, mot_float_type_dtype)).itemsize
        return [cl.LocalMemory(itemsize * self._size_func(workgroup_size))]

    def get_nmr_kernel_inputs(self):
        return 1


class Array(KernelData):

    def __init__(self, data, ctype=None, as_scalar=False, parallelize_over_first_dimension=True,
                 mode='rw', use_host_ptr=True):
        """Loads the given array as a buffer into one or more OpenCL contexts.

        By default, this expects multi-dimensional arrays (n, m, k, ...) which holds a (m, k, ...) for every data
        point ``n``. As such, every global work item will only receive a subset of the matrix. If every work item
        needs access to all items, set parallelize_over_first_dimension to False.

        In general, this class uses the provided data's dtype for the buffering. If the ctype is provided and the
        data's dtype does not match the ctype, we will convert the data to the righ dtype. If the ctype is set
        to mot_float_type and the mot_float_type changes, then a new Array data class is returned.

        Args:
            data (ndarray): the data to load in the kernel
            ctype (str): the desired c-type for in use in the kernel, like ``int``, ``float`` or ``mot_float_type``.
                If None it is implied from the provided data. If not matching the data, we convert the data.
            as_scalar (boolean): if given and if the data is only a 1d, we will load the value as a scalar in the
                data struct. As such, one does not need to evaluate as a pointer.
            parallelize_over_first_dimension (boolean): only applicable for multi-dimensional arrays (n, m, k, ...)
                where ``n`` is expected to correspond to problem instances. If True, we will load the data as
                (m, k, ...) arrays for each problem instance. If False, the data will be loaded as is, and each problem
                instance will have a reference to the complete array.
            mode (str): one of 'r', 'w' or 'rw', for respectively read, write or read and write.
                This defines how the device is planned on accessing this array, and, defines if we will read the
                data back after applying a kernel (if 'w' is included we will write data back, else, not).
            use_host_ptr (boolean): if set, we will use the USE_HOST_PTR flag and use map/unmap for data transfers
                if not set, we create a device side buffer and use explicit read and write commands to transfer the data
        """
        if isinstance(data, (list, tuple)):
            data = np.array(data)

        self._mode = mode
        self._is_readable = 'r' in mode
        self._is_writable = 'w' in mode

        self._data = data
        if ctype and not ctype.startswith('mot_float_type'):
            self._data = convert_data_to_dtype(self._data, ctype)

        self._ctype = ctype or dtype_to_ctype(self._data.dtype)
        self._mot_float_dtype = None
        self._backup_data_reference = None
        self._as_scalar = as_scalar
        self._parallelize_over_first_dimension = parallelize_over_first_dimension

        self._use_host_ptr = use_host_ptr
        self._buffer_cache = {}  # caching the buffers per context

        self._data_length = 1
        if len(self._data.shape):
            self._data_length = self._data.strides[0] // self._data.itemsize
        if not self._parallelize_over_first_dimension:
            self._data_length = self._data.size

        if self._as_scalar and len(np.squeeze(self._data).shape) > 1:
            raise ValueError('The option "as_scalar" was set, but the data has more than one dimensions.')

    @property
    def ctype(self):
        return self._ctype

    @property
    def mode(self):
        """Get the read write mode defined for this array.

        Returns:
            str: the mode in which the buffer should be created, a string like 'r', 'w' or 'rw'
        """
        return self._mode

    def get_subset(self, problem_indices=None, batch_range=None):
        if problem_indices is None and batch_range is None:
            return self
        if not self._parallelize_over_first_dimension:
            return self

        if problem_indices is None:
            return Array(self._data[batch_range[0]:batch_range[1]], ctype=self._ctype,
                         mode=self._mode, as_scalar=self._as_scalar,
                         parallelize_over_first_dimension=self._parallelize_over_first_dimension,
                         use_host_ptr=self._use_host_ptr)

        def is_consecutive(l):
            return np.sum(np.diff(np.sort(l)) == 1) >= (len(l) - 1)

        if is_consecutive(problem_indices):
            return Array(self._data[problem_indices[0]:(problem_indices[-1] + 1)], ctype=self._ctype,
                         mode=self._mode, as_scalar=self._as_scalar,
                         parallelize_over_first_dimension=self._parallelize_over_first_dimension,
                         use_host_ptr=self._use_host_ptr)

        return Array(self._data[problem_indices], ctype=self._ctype,
                     mode=self._mode, as_scalar=self._as_scalar,
                     parallelize_over_first_dimension=self._parallelize_over_first_dimension,
                     use_host_ptr=self._use_host_ptr)

    def set_mot_float_dtype(self, mot_float_dtype):
        self._mot_float_dtype = mot_float_dtype

        if self._ctype.startswith('mot_float_type'):
            if self._backup_data_reference is not None:
                self._data = self._backup_data_reference
                self._backup_data_reference = None

            new_data = convert_data_to_dtype(self._data, self._ctype,
                                             mot_float_type=dtype_to_ctype(mot_float_dtype))

            if new_data is not self._data:
                self._backup_data_reference = self._data
                self._data = new_data
                self._buffer_cache = {}  # cache is invalidated

        # data length may change when an CL vector type is converted from (n, 3) shape to (n,)
        self._data_length = 1
        if len(self._data.shape):
            self._data_length = self._data.strides[0] // self._data.itemsize
        if not self._parallelize_over_first_dimension:
            self._data_length = self._data.size

    def get_data(self):
        return self._data

    def get_children(self):
        return []

    def get_scalar_arg_dtypes(self):
        return [None]

    def get_type_definitions(self):
        return ''

    def initialize_variable(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        if not self._as_scalar:
            if address_space == 'private':
                return '''
                    private {ctype} {v_name}[{nmr_elements}];

                    for(uint i = 0; i < {nmr_elements}; i++){{
                        {v_name}[i] = {k_name}[{offset} + i];
                    }}
                '''.format(ctype=self._ctype, v_name=variable_name, k_name=kernel_param_name,
                           nmr_elements=self._data_length,
                           offset=self._get_offset_str(problem_id_substitute))
            elif address_space == 'local':
                return '''
                    local {ctype} {v_name}[{nmr_elements}];

                    if(get_local_id(0) == 0){{
                        for(uint i = 0; i < {nmr_elements}; i++){{
                            {v_name}[i] = {k_name}[{offset} + i];
                        }}
                    }}
                    barrier(CLK_LOCAL_MEM_FENCE);
                '''.format(ctype=self._ctype, v_name=variable_name, k_name=kernel_param_name,
                           nmr_elements=self._data_length,
                           offset=self._get_offset_str(problem_id_substitute))
        return ''

    def get_function_call_input(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        if self._as_scalar:
            return '{}[{}]'.format(kernel_param_name, self._get_offset_str(problem_id_substitute))
        else:
            if address_space == 'global':
                return '{} + {}'.format(kernel_param_name, self._get_offset_str(problem_id_substitute))
            elif address_space == 'private':
                return variable_name
            elif address_space == 'local':
                return variable_name

    def post_function_callback(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        if self._is_writable:
            if not self._as_scalar:
                if address_space == 'private':
                    return '''
                        for(uint i = 0; i < {nmr_elements}; i++){{
                            {k_name}[{offset} + i] = {v_name}[i];
                        }}
                    '''.format(v_name=variable_name, k_name=kernel_param_name,
                               nmr_elements=self._data_length, offset=self._get_offset_str(problem_id_substitute))
                elif address_space == 'local':
                    return '''
                        if(get_local_id(0) == 0){{
                            for(uint i = 0; i < {nmr_elements}; i++){{
                                {k_name}[{offset} + i] = {v_name}[i];
                            }}
                        }}
                    '''.format(v_name=variable_name, k_name=kernel_param_name,
                               nmr_elements=self._data_length,
                               offset=self._get_offset_str(problem_id_substitute))
        return ''

    def get_struct_declaration(self, name):
        if self._as_scalar:
            return '{} {};'.format(self._ctype, name)
        return 'global {}* restrict {};'.format(self._ctype, name)

    def get_struct_initialization(self, variable_name, kernel_param_name, problem_id_substitute):
        return self.get_function_call_input(variable_name, kernel_param_name, problem_id_substitute, 'global')

    def get_kernel_parameters(self, kernel_param_name):
        return ['global {}* restrict {}'.format(self._ctype, kernel_param_name)]

    def enqueue_host_access(self, cl_environments, is_blocking=True, wait_for=None):
        if isinstance(cl_environments, CLEnvironment):
            cl_environments = [cl_environments]

        for env in cl_environments:
            self.get_kernel_inputs(env, 1)

        wait_for = wait_for or {}
        events = {}

        if self._is_writable:
            for env in cl_environments:
                context = env.context

                wait_list = []
                for wait_env, wait_event in wait_for.items():
                    if wait_env.context is context:
                        wait_list.append(wait_event)

                if not any(e.context is env.context for e in events.keys()):
                    if self._use_host_ptr:
                        _, event = cl.enqueue_map_buffer(
                            env.queue, self._buffer_cache[context],
                            cl.map_flags.READ, 0, self._data.shape, self._data.dtype,
                            order="C", wait_for=wait_list, is_blocking=False)
                    else:
                        event = cl.enqueue_copy(env.queue, self._data, self._buffer_cache[context],
                                                is_blocking=False, wait_for=wait_list)

                    events[env] = event

        if is_blocking:
            for env in cl_environments:
                env.queue.finish()

        return events

    def enqueue_device_access(self, cl_environments, is_blocking=True, wait_for=None):
        if isinstance(cl_environments, CLEnvironment):
            cl_environments = [cl_environments]

        for env in cl_environments:
            self.get_kernel_inputs(env, 1)

        wait_for = wait_for or {}
        events = {}

        if not self._use_host_ptr:
            for env in cl_environments:
                context = env.context

                wait_list = []
                for wait_env, wait_event in wait_for.items():
                    if wait_env.context is context:
                        wait_list.append(wait_event)

                if not any(e.context is env.context for e in events.keys()):
                    event = cl.enqueue_copy(env.queue, self._buffer_cache[context], self._data,
                                            is_blocking=False, wait_for=wait_list)
                    events[env] = event

        if is_blocking:
            for env in cl_environments:
                env.queue.finish()

        return events

    def get_kernel_inputs(self, cl_environment, workgroup_size):
        def get_mem_flags():
            if self._is_writable:
                if self._is_readable:
                    return cl.mem_flags.READ_WRITE
                else:
                    return cl.mem_flags.WRITE_ONLY
            else:
                return cl.mem_flags.READ_ONLY

        cl_context = cl_environment.context

        if cl_context not in self._buffer_cache:
            if self._use_host_ptr:
                self._buffer_cache[cl_context] = cl.Buffer(cl_context,
                                                           get_mem_flags() | cl.mem_flags.USE_HOST_PTR,
                                                           hostbuf=self._data)
            else:
                self._buffer_cache[cl_context] = cl.Buffer(cl_context, get_mem_flags(), size=self._data.nbytes)

        return [self._buffer_cache[cl_context]]

    def get_nmr_kernel_inputs(self):
        return 1

    def _get_offset_str(self, problem_id_substitute):
        if self._parallelize_over_first_dimension:
            offset_str = str(self._data_length) + ' * {problem_id}'
        else:
            offset_str = '0'
        return offset_str.replace('{problem_id}', problem_id_substitute)


class Zeros(KernelData):

    def __init__(self, shape, ctype, parallelize_over_first_dimension=True, host_accessible=True, mode='rw'):
        """Allocate an output buffer of the given shape.

        This is meant to quickly allocate a buffer large enough to hold the data requested. After running an OpenCL
        kernel you can get the written data using the method :meth:`get_data`.

        If host accessible is False and you call the method :meth:`get_subset`, the resulting computations will
        be split over multiple buffers and there is no single continuous buffer with all the values. Therefore,
        this flag is not usable if you intend to share a :class:`Zeros` with multiple CLEnvironments.

        Args:
            shape (int or tuple): the shape of the output array
            ctype (str): the desired C-type for this zero's array
            parallelize_over_first_dimension (boolean): only applicable for multi-dimensional arrays (n, m, k, ...)
                where ``n`` is expected to correspond to problem instances. If True, we will load the data as
                (m, k, ...) arrays for each problem instance. If False, the data will be loaded as is, and each problem
                instance will have a reference to the complete array.
            host_accessible (boolean): if the array needs to be accessible from the host. If not, we can allocate
                the buffer on the device, saving time on memory copies.
            mode (str): one of 'r', 'w' or 'rw', for respectively read, write or read and write. This sets the
                mode of how the data is loaded into the compute device's memory.
        """
        self._shape = shape
        if isinstance(self._shape, numbers.Number):
            self._shape = (self._shape,)
        self._host_accessible = host_accessible
        self._ctype = ctype

        self._mode = mode
        self._is_readable = 'r' in mode
        self._is_writable = 'w' in mode

        self._mot_float_dtype = None
        self._parallelize_over_first_dimension = parallelize_over_first_dimension

        self._buffer_cache = {}  # caching the buffers per context

        self._data_length = 1
        if isinstance(shape, (list, tuple)) and len(shape) > 1:
            self._data_length = np.prod(shape[1:])
        if not self._parallelize_over_first_dimension:
            self._data_length = np.prod(shape)

        if host_accessible:
            self._array = Array(np.zeros(shape, dtype=ctype_to_dtype(ctype)), ctype,
                                parallelize_over_first_dimension=parallelize_over_first_dimension,
                                mode=mode, as_scalar=False)

    @property
    def ctype(self):
        return self._ctype

    @property
    def mode(self):
        return self._mode

    def get_subset(self, problem_indices=None, batch_range=None):
        if self._host_accessible:
            return self._array.get_subset(problem_indices, batch_range)
        if problem_indices is not None:
            shape = (len(problem_indices),) + self._shape[1:]
        else:
            shape = (batch_range[1] - batch_range[0],) + self._shape[1:]
        return Zeros(shape, self._ctype, mode=self._mode,
                     parallelize_over_first_dimension=self._parallelize_over_first_dimension,
                     host_accessible=False)

    def set_mot_float_dtype(self, mot_float_dtype):
        if self._host_accessible:
            return self._array.set_mot_float_dtype(mot_float_dtype)
        self._mot_float_dtype = mot_float_dtype

    def get_data(self):
        if self._host_accessible:
            return self._array.get_data()
        raise ValueError('Can not get the data from a non host accessible array.')

    def get_children(self):
        return []

    def get_scalar_arg_dtypes(self):
        return [None]

    def get_type_definitions(self):
        return ''

    def initialize_variable(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        if self._host_accessible:
            return self._array.initialize_variable(variable_name, kernel_param_name,
                                                   problem_id_substitute, address_space)

        if address_space == 'private':
            return '''
                private {ctype} {v_name}[{nmr_elements}];

                for(uint i = 0; i < {nmr_elements}; i++){{
                    {v_name}[i] = {k_name}[{offset} + i];
                }}
            '''.format(ctype=self._ctype, v_name=variable_name, k_name=kernel_param_name,
                       nmr_elements=self._data_length,
                       offset=self._get_offset_str(problem_id_substitute))
        elif address_space == 'local':
            return '''
                local {ctype} {v_name}[{nmr_elements}];

                if(get_local_id(0) == 0){{
                    for(uint i = 0; i < {nmr_elements}; i++){{
                        {v_name}[i] = {k_name}[{offset} + i];
                    }}
                }}
                barrier(CLK_LOCAL_MEM_FENCE);
            '''.format(ctype=self._ctype, v_name=variable_name, k_name=kernel_param_name,
                       nmr_elements=self._data_length,
                       offset=self._get_offset_str(problem_id_substitute))

    def get_function_call_input(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        if self._host_accessible:
            return self._array.get_function_call_input(variable_name, kernel_param_name,
                                                       problem_id_substitute, address_space)

        if address_space == 'global':
            return '{} + {}'.format(kernel_param_name, self._get_offset_str(problem_id_substitute))
        elif address_space == 'private':
            return variable_name
        elif address_space == 'local':
            return variable_name

    def post_function_callback(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        if self._host_accessible:
            return self._array.post_function_callback(variable_name, kernel_param_name,
                                                      problem_id_substitute, address_space)

        if self._is_writable:
            if address_space == 'private':
                return '''
                    for(uint i = 0; i < {nmr_elements}; i++){{
                        {k_name}[{offset} + i] = {v_name}[i];
                    }}
                '''.format(v_name=variable_name, k_name=kernel_param_name,
                           nmr_elements=self._data_length, offset=self._get_offset_str(problem_id_substitute))
            elif address_space == 'local':
                return '''
                    if(get_local_id(0) == 0){{
                        for(uint i = 0; i < {nmr_elements}; i++){{
                            {k_name}[{offset} + i] = {v_name}[i];
                        }}
                    }}
                '''.format(v_name=variable_name, k_name=kernel_param_name,
                           nmr_elements=self._data_length,
                           offset=self._get_offset_str(problem_id_substitute))
        return ''

    def get_struct_declaration(self, name):
        if self._host_accessible:
            return self._array.get_struct_declaration(name)

        return 'global {}* restrict {};'.format(self._ctype, name)

    def get_struct_initialization(self, variable_name, kernel_param_name, problem_id_substitute):
        if self._host_accessible:
            return self._array.get_struct_initialization(variable_name, kernel_param_name, problem_id_substitute)
        return self.get_function_call_input(variable_name, kernel_param_name, problem_id_substitute, 'global')

    def get_kernel_parameters(self, kernel_param_name):
        return ['global {}* restrict {}'.format(self._ctype, kernel_param_name)]

    def enqueue_host_access(self, cl_environments, is_blocking=True, wait_for=None):
        if self._host_accessible:
            return self._array.enqueue_host_access(cl_environments, is_blocking=is_blocking, wait_for=wait_for)
        return {}

    def enqueue_device_access(self, cl_environments, is_blocking=True, wait_for=None):
        return {}

    def get_kernel_inputs(self, cl_environment, workgroup_size):
        cl_context = cl_environment.context

        if self._host_accessible:
            return self._array.get_kernel_inputs(cl_environment, workgroup_size)

        mot_float_type_dtype = None
        if self._mot_float_dtype:
            mot_float_type_dtype = dtype_to_ctype(self._mot_float_dtype)
        dtype = np.dtype(ctype_to_dtype(self._ctype, mot_float_type_dtype))
        itemsize = dtype.itemsize

        if cl_context not in self._buffer_cache:
            if self._is_writable:
                if self._is_readable:
                    flags = cl.mem_flags.READ_WRITE
                else:
                    flags = cl.mem_flags.WRITE_ONLY
            else:
                flags = cl.mem_flags.READ_ONLY

            buffer = cl.Buffer(cl_context, flags, size=np.prod(self._shape) * itemsize)

            cl.enqueue_fill_buffer(cl_environment.queue, buffer,
                                   np.zeros(1, dtype=dtype), 0, np.prod(self._shape) * itemsize)

            self._buffer_cache[cl_context] = buffer

        return [self._buffer_cache[cl_context]]

    def get_nmr_kernel_inputs(self):
        return 1

    def _get_offset_str(self, problem_id_substitute):
        if self._parallelize_over_first_dimension:
            offset_str = str(self._data_length) + ' * {problem_id}'
        else:
            offset_str = '0'
        return offset_str.replace('{problem_id}', problem_id_substitute)


class CompositeArray(KernelData):

    def __init__(self, elements, ctype, address_space='private'):
        """An array filled with the given kernel data elements.

        Each of the given elements should be a :class:`Scalar` or an :class:`Array` with the property `as_scalar`
        set to True. We will load each value of the given elements into a private array.

        Args:
            elements (List[KernelData]): the kernel data elements to load into the private array
            ctype (str): the data type of this structure
            address_space (str): the address space for the allocation of the main array
        """
        self._elements = elements
        self._ctype = ctype
        self._address_space = address_space

        if self._address_space == 'private':
            self._composite_array = PrivateMemory(len(self._elements), self._ctype)
        elif self._address_space == 'local':
            self._composite_array = LocalMemory(self._ctype, len(self._elements))
        elif self._address_space == 'global':
            self._composite_array = Zeros(len(self._elements), self._ctype, mode='rw',
                                          parallelize_over_first_dimension=False)

    @property
    def ctype(self):
        return self._ctype

    def get_subset(self, problem_indices=None, batch_range=None):
        return self

    def set_mot_float_dtype(self, mot_float_dtype):
        for element in self._elements:
            element.set_mot_float_dtype(mot_float_dtype)
        self._composite_array.set_mot_float_dtype(mot_float_dtype)

    def get_data(self):
        return [item.get_data() for item in self._elements]

    def get_children(self):
        return self._elements

    def get_scalar_arg_dtypes(self):
        dtypes = list(self._composite_array.get_scalar_arg_dtypes())
        for d in self._elements:
            dtypes.extend(d.get_scalar_arg_dtypes())
        return dtypes

    def get_type_definitions(self):
        return '\n'.join(element.get_type_definitions() for element in self._elements)

    def initialize_variable(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return_str = self._composite_array.initialize_variable(variable_name, kernel_param_name,
                                                               problem_id_substitute, address_space)

        for ind, data in enumerate(self._elements):
            return_str += data.initialize_variable('{}_{}'.format(variable_name, str(ind)),
                                                   '{}_{}'.format(kernel_param_name, str(ind)),
                                                   problem_id_substitute, 'global')

            return_str += '{}[{}] = {};\n'.format(
                kernel_param_name,
                ind,
                data.get_struct_initialization(
                    '{}_{}'.format(variable_name, str(ind)),
                    '{}_{}'.format(kernel_param_name, str(ind)), problem_id_substitute))

        return return_str

    def get_function_call_input(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return self._composite_array.get_function_call_input(variable_name, kernel_param_name,
                                                             problem_id_substitute, address_space)

    def post_function_callback(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return self._composite_array.post_function_callback(variable_name, kernel_param_name,
                                                            problem_id_substitute, address_space)

    def get_struct_declaration(self, name):
        return self._composite_array.get_struct_declaration(name)

    def get_struct_initialization(self, variable_name, kernel_param_name, problem_id_substitute):
        return self._composite_array.get_struct_initialization(variable_name, kernel_param_name, problem_id_substitute)

    def get_kernel_parameters(self, kernel_param_name):
        parameters = list(self._composite_array.get_kernel_parameters(kernel_param_name))
        for ind, d in enumerate(self._elements):
            parameters.extend(d.get_kernel_parameters('{}_{}'.format(kernel_param_name, str(ind))))
        return parameters

    def enqueue_host_access(self, cl_environments, is_blocking=True, wait_for=None):
        events = {}
        for d in self._elements:
            events.update(d.enqueue_host_access(cl_environments, is_blocking=is_blocking, wait_for=wait_for))
        return events

    def enqueue_device_access(self, cl_environments, is_blocking=True, wait_for=None):
        events = {}
        for d in self._elements:
            events.update(d.enqueue_device_access(cl_environments, is_blocking=is_blocking, wait_for=wait_for))
        return events

    def get_kernel_inputs(self, cl_environment, workgroup_size):
        data = list(self._composite_array.get_kernel_inputs(cl_environment, workgroup_size))
        for d in self._elements:
            data.extend(d.get_kernel_inputs(cl_environment, workgroup_size))
        return data

    def get_nmr_kernel_inputs(self):
        return self._composite_array.get_nmr_kernel_inputs() + \
               sum(element.get_nmr_kernel_inputs() for element in self._elements)
