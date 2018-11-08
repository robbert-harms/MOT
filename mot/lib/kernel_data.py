import numbers
from collections import OrderedDict, Mapping

import numpy as np
import pyopencl as cl

from mot.lib.utils import dtype_to_ctype, ctype_to_dtype, convert_data_to_dtype, is_vector_ctype, split_vector_ctype

__author__ = 'Robbert Harms'
__date__ = '2018-04-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class KernelData:

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

    def enqueue_readouts(self, queue, buffers, range_start, range_end):
        """Enqueue readouts for this kernel input data object.

        This should add non-blocking readouts to the given queue.

        Args:
            queue (opencl queue): the queue on which to add the unmap buffer command
            buffers (List[pyopencl._cl.Buffer.Buffer]): the list of buffers corresponding to this kernel data.
                These buffers are obtained earlier from the method :meth:`get_kernel_inputs`.
            range_start (int): the start of the range to read out (in the first dimension)
            range_end (int): the end of the range to read out (in the first dimension)
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
             address_space (str): the desired address space for this variable, defined by the parameter of the called
                function.

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

    def get_kernel_inputs(self, cl_context, workgroup_size):
        """Get the kernel input data matching the list of parameters of :meth:`get_kernel_parameters`.

        Since the kernels follow the map/unmap paradigm make sure to use the ``USE_HOST_PTR`` when making
        writable data objects.

        Args:
            cl_context (pyopencl.Context): the CL context in which we are working.
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

    def set_mot_float_dtype(self, mot_float_dtype):
        for element in self._elements.values():
            element.set_mot_float_dtype(mot_float_dtype)

    def get_data(self):
        data = {}
        for name, value in self._elements.items():
            data[name] = value.get_data()
        return data

    def get_scalar_arg_dtypes(self):
        dtypes = []
        for d in self._elements.values():
            dtypes.extend(d.get_scalar_arg_dtypes())
        return dtypes

    def enqueue_readouts(self, queue, buffers, range_start, range_end):
        buffer_ind = 0

        for d in self._elements.values():
            if d.get_nmr_kernel_inputs():
                d.enqueue_readouts(queue, buffers[buffer_ind:buffer_ind + d.get_nmr_kernel_inputs()],
                                   range_start, range_end)
                buffer_ind += d.get_nmr_kernel_inputs()

    def get_type_definitions(self):
        other_structs = '\n'.join(element.get_type_definitions() for element in self._elements.values())

        if self._anonymous:
            return other_structs

        return other_structs + '''
            typedef struct {ctype}{{
                {definitions}
            }} {ctype};
        '''.format(ctype=self._ctype,
                   definitions='\n'.join(data.get_struct_declaration(name) for name, data in self._elements.items()))

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

    def get_kernel_inputs(self, cl_context, workgroup_size):
        data = []
        for d in self._elements.values():
            data.extend(d.get_kernel_inputs(cl_context, workgroup_size))
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

    def __init__(self, value, ctype=None):
        """A kernel input scalar.

        This will insert the given value directly into the kernel's source code, and will not load it as a buffer.

        Args:
            value (number): the number to insert into the kernel as a scalar.
            ctype (str): the desired c-type for in use in the kernel, like ``int``, ``float`` or ``mot_float_type``.
                If None it is implied from the value.
        """
        if isinstance(value, str) and value == 'INFINITY':
            self._value = np.inf
        elif isinstance(value, str) and value == '-INFINITY':
            self._value = -np.inf
        else:
            self._value = np.array(value)
        self._ctype = ctype or dtype_to_ctype(self._value.dtype)
        self._mot_float_dtype = None

    def get_data(self):
        if self._ctype.startswith('mot_float_type'):
            return np.asscalar(self._value.astype(self._mot_float_dtype))
        return np.asscalar(self._value)

    def get_scalar_arg_dtypes(self):
        return []

    def enqueue_readouts(self, queue, buffers, range_start, range_end):
        pass

    def get_type_definitions(self):
        return ''

    def get_struct_declaration(self, name):
        return '{} {};'.format(self._ctype, name)

    def get_struct_initialization(self, variable_name, kernel_param_name, problem_id_substitute):
        return self.get_function_call_input(variable_name, kernel_param_name, problem_id_substitute, 'private')

    def get_kernel_parameters(self, kernel_param_name):
        return []

    def get_kernel_inputs(self, cl_context, workgroup_size):
        return []

    def get_nmr_kernel_inputs(self):
        return 0

    def set_mot_float_dtype(self, mot_float_dtype):
        self._mot_float_dtype = mot_float_dtype

    def initialize_variable(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
        return ''

    def get_function_call_input(self, variable_name, kernel_param_name, problem_id_substitute, address_space):
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

    def set_mot_float_dtype(self, mot_float_dtype):
        self._mot_float_dtype = mot_float_dtype

    def get_data(self):
        return None

    def get_scalar_arg_dtypes(self):
        return [None]

    def enqueue_readouts(self, queue, buffers, range_start, range_end):
        pass

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

    def get_kernel_inputs(self, cl_context, workgroup_size):
        itemsize = np.dtype(ctype_to_dtype(self._ctype, dtype_to_ctype(self._mot_float_dtype))).itemsize
        return [cl.LocalMemory(itemsize * self._size_func(workgroup_size))]

    def get_nmr_kernel_inputs(self):
        return 1


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

    def set_mot_float_dtype(self, mot_float_dtype):
        self._mot_float_dtype = mot_float_dtype

    def get_data(self):
        return None

    def get_scalar_arg_dtypes(self):
        return []

    def enqueue_readouts(self, queue, buffers, range_start, range_end):
        pass

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

    def get_kernel_inputs(self, cl_context, workgroup_size):
        return []

    def get_nmr_kernel_inputs(self):
        return 0


class Array(KernelData):

    def __init__(self, data, ctype=None, mode='r', offset_str=None, ensure_zero_copy=False, as_scalar=False):
        """Loads the given array as a buffer into the kernel.

        By default, this will try to offset the data in the kernel by the stride of the first dimension multiplied
        with the problem id by the kernel. For example, if a (n, m) matrix is provided, this will offset the data
        by ``{problem_id} * m``.

        This class will adapt the data to match the ctype (if necessary) and it might copy the data as a consecutive
        array for direct memory access by the CL environment. Depending on those transformations, a copy of the original
        array may be made. As such, if ``is_writable`` would have been set, the return values might be written to
        a different array. To retrieve the output data after kernel execution, use the method :meth:`get_data`.
        Alternatively, set ``ensure_zero_copy`` to True, this ensures that the return values are written to the
        same reference by raising a ValueError if the data has to be copied to be used in the kernel.

        Args:
            data (ndarray): the data to load in the kernel
            ctype (str): the desired c-type for in use in the kernel, like ``int``, ``float`` or ``mot_float_type``.
                If None it is implied from the provided data.
            mode (str): one of 'r', 'w' or 'rw', for respectively read, write or read and write. This sets the
                mode of how the data is loaded into the compute device's memory.
            offset_str (str): the offset definition, can use ``{problem_id}`` for multiplication purposes. Set to 0
                for no offset.
            ensure_zero_copy (boolean): only used if ``is_writable`` is set to True. If set, we guarantee that the
                return values are written to the same input array. This allows the user of this class to user their
                reference to the underlying data, relieving the user of having to use :meth:`get_data`.
            as_scalar (boolean): if given and if the data is only a 1d, we will load the value as a scalar in the
                data struct. As such, one does not need to evaluate as a pointer.
        """
        self._is_readable = 'r' in mode
        self._is_writable = 'w' in mode

        self._requirements = ['C', 'A', 'O']
        if self._is_writable:
            self._requirements.append('W')

        self._data = np.require(data, requirements=self._requirements)
        if ctype and not ctype.startswith('mot_float_type'):
            self._data = convert_data_to_dtype(self._data, ctype)

        self._offset_str = offset_str
        self._ctype = ctype or dtype_to_ctype(self._data.dtype)
        self._mot_float_dtype = None
        self._backup_data_reference = None
        self._ensure_zero_copy = ensure_zero_copy
        self._as_scalar = as_scalar

        self._data_length = 1
        if len(self._data.shape):
            self._data_length = self._data.strides[0] // self._data.itemsize
        if self._offset_str == '0' or self._offset_str == 0:
            self._data_length = self._data.size

        if self._as_scalar and len(np.squeeze(self._data).shape) > 1:
            raise ValueError('The option "as_scalar" was set, but the data has more than one dimensions.')

        if self._is_writable and self._ensure_zero_copy and self._data is not data:
            raise ValueError('Zero copy was set but we had to make '
                             'a copy to guarantee the writing and ctype requirements.')

    def set_mot_float_dtype(self, mot_float_dtype):
        self._mot_float_dtype = mot_float_dtype

        if self._ctype.startswith('mot_float_type'):
            if self._backup_data_reference is not None:
                self._data = self._backup_data_reference
                self._backup_data_reference = None

            new_data = convert_data_to_dtype(self._data, self._ctype,
                                             mot_float_type=dtype_to_ctype(mot_float_dtype))
            new_data = np.require(new_data, requirements=self._requirements)

            if new_data is not self._data:
                self._backup_data_reference = self._data
                self._data = new_data

                if self._is_writable and self._ensure_zero_copy:
                    raise ValueError('We had to make a copy of the data while zero copy was set to True.')

        # data length may change when an CL vector type is converted from (n, 3) shape to (n,)
        self._data_length = 1
        if len(self._data.shape):
            self._data_length = self._data.strides[0] // self._data.itemsize
        if self._offset_str == '0' or self._offset_str == 0:
            self._data_length = self._data.size

    def get_data(self):
        return self._data

    def get_scalar_arg_dtypes(self):
        return [None]

    def enqueue_readouts(self, queue, buffers, range_start, range_end):
        if self._is_writable:
            nmr_problems = int(range_end - range_start)
            cl.enqueue_map_buffer(
                queue, buffers[0], cl.map_flags.READ,
                int(range_start * self._data.strides[0]),
                (nmr_problems,) + self._data.shape[1:], self._data.dtype,
                order="C", wait_for=None, is_blocking=False)

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

    def get_kernel_inputs(self, cl_context, workgroup_size):
        if self._is_writable:
            if self._is_readable:
                flags = cl.mem_flags.READ_WRITE
            else:
                flags = cl.mem_flags.WRITE_ONLY
        else:
            flags = cl.mem_flags.READ_ONLY

        return [cl.Buffer(cl_context, flags | cl.mem_flags.USE_HOST_PTR, hostbuf=self._data)]

    def get_nmr_kernel_inputs(self):
        return 1

    def _get_offset_str(self, problem_id_substitute):
        if self._offset_str is None:
            offset_str = str(self._data_length) + ' * {problem_id}'
        else:
            offset_str = str(self._offset_str)
        return offset_str.replace('{problem_id}', problem_id_substitute)


class Zeros(Array):

    def __init__(self, shape, ctype, offset_str=None, mode='w'):
        """Allocate an output buffer of the given shape.

        This is meant to quickly allocate a buffer large enough to hold the data requested. After running an OpenCL
        kernel you can get the written data using the method :meth:`get_data`.

        Args:
            shape (int or tuple): the shape of the output array
            offset_str (str): the offset definition, can use ``{problem_id}`` for multiplication purposes. Set to 0
                for no offset.
            mode (str): one of 'r', 'w' or 'rw', for respectively read, write or read and write. This sets the
                mode of how the data is loaded into the compute device's memory.
        """
        super().__init__(np.zeros(shape, dtype=ctype_to_dtype(ctype)), ctype, offset_str=offset_str,
                         mode=mode, as_scalar=False)


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
            self._composite_array = Zeros(len(self._elements), self._ctype, offset_str='0', mode='rw')

    def set_mot_float_dtype(self, mot_float_dtype):
        for element in self._elements:
            element.set_mot_float_dtype(mot_float_dtype)
        self._composite_array.set_mot_float_dtype(mot_float_dtype)

    def get_data(self):
        return [item.get_data() for item in self._elements]

    def get_scalar_arg_dtypes(self):
        dtypes = list(self._composite_array.get_scalar_arg_dtypes())
        for d in self._elements:
            dtypes.extend(d.get_scalar_arg_dtypes())
        return dtypes

    def enqueue_readouts(self, queue, buffers, range_start, range_end):
        pass

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

    def get_kernel_inputs(self, cl_context, workgroup_size):
        data = list(self._composite_array.get_kernel_inputs(cl_context, workgroup_size))
        for d in self._elements:
            data.extend(d.get_kernel_inputs(cl_context, workgroup_size))
        return data

    def get_nmr_kernel_inputs(self):
        return self._composite_array.get_nmr_kernel_inputs() + \
               sum(element.get_nmr_kernel_inputs() for element in self._elements)
