import numpy as np
import pyopencl as cl

from mot.cl_data_type import SimpleCLDataType
from mot.utils import dtype_to_ctype, ctype_to_dtype, convert_data_to_dtype

__author__ = 'Robbert Harms'
__date__ = '2018-04-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class KernelData(object):

    def set_mot_float_dtype(self, mot_float_dtype):
        """Set the numpy data type corresponding to the ``mot_float_type`` ctype.

        This is set by the kernel input data manager and is meant to update the state of the input data according
        to this setting.

        Args:
            mot_float_dtype (dtype): the numpy data type that is to correspond with the ``mot_float_type`` used in the
                kernels.
        """
        raise NotImplementedError()

    @property
    def is_scalar(self):
        """Check if the implemented input data is a scalar or not.

        Since scalars are loaded differently as buffers in the kernel, we have to check if this data is a scalar or not.

        Returns:
            boolean: if the implemented type should be loaded as a scalar or not
        """
        raise NotImplementedError()

    @property
    def read_data_back(self):
        """Check if this input data should be read back after kernel execution.

        Returns:
            boolean: if, after kernel launch, the data should be mapped from the compute device back to host memory.
        """
        raise NotImplementedError()

    def get_scalar_arg_dtype(self):
        """Get the numpy data type we should report in the kernel call for this input element.

        If we are inserting scalars in the kernel we need to provide the CL runtime with the correct data type
        of the function.

        Args:
            mot_float_dtype (dtype): the data type of the mot_float_type

        Returns:
            dtype: the numpy data type for this element, or None if this is not a scalar.
        """
        raise NotImplementedError()

    def get_data(self):
        """Get the underlying data.

        This should return the current state of the data object.

        Returns:
            the underlying data object, make sure this is of your desired data type. Can return None if
                this input data has no actual data.
        """
        raise NotImplementedError()

    def get_struct_declaration(self, name):
        """Get the declaration of this input data in the ``mot_data_struct`` object.

        Args:
            name (str): the name for this data, i.e. how it is represented in the kernel.

        Returns:
            str: the declaration of this input data in the kernel data struct
        """
        raise NotImplementedError()

    def get_kernel_argument_declaration(self, name):
        """Get the kernel argument declaration of this parameter.

        Args:
            name (str): the name for this data, i.e. how it is represented in the kernel.

        Returns:
            str: the kernel argument declaration
        """
        raise NotImplementedError()

    def get_struct_init_string(self, name, problem_id_substitute):
        """Create the structure initialization string.

        Args:
            name (str): the name of this data in the kernel
            problem_id_substitute (str): the substitute for the ``{problem_id}`` in the kernel data info elements.

        Returns:
            str: the instantiation string for the data struct of this input data
        """
        raise NotImplementedError()

    def get_kernel_inputs(self, cl_context, workgroup_size):
        """Get the kernel CL input object.

        Since the kernels follow the map/unmap paradigm make sure to use the ``USE_HOST_PTR`` when making
        writable data objects.

        Args:
            cl_context (pyopencl.Context): the CL context in which we are working.
            workgroup_size (int or None): the workgroup size the kernel will use. If None, we are not using any
                local reduction.

        Returns:
            a buffer, a local memory object, a scalars, etc., anything that can be loaded into the kernel.
        """
        raise NotImplementedError()

    def include_in_kernel_call(self):
        """Check if this data needs to be included in the kernel arguments.

        Returns:
            boolean: if the corresponding data needs to be included in the kernel call arguments. If set to False
                we typically expect the data to be inlined in the kernel. If set to True, we add the data to the
                kernel arguments.
        """
        raise NotImplementedError()


class KernelScalar(KernelData):

    def __init__(self, value, ctype=None):
        """A kernel input scalar.

        Args:
            value (number): the number to insert into the kernel as a scalar.
            ctype (str): the desired c-type for in use in the kernel, like ``int``, ``float`` or ``mot_float_type``.
                If None it is implied from the value.
        """
        self._value = np.array(value)
        self._ctype = ctype or dtype_to_ctype(self._value.dtype)
        self._mot_float_dtype = None

    def set_mot_float_dtype(self, mot_float_dtype):
        self._mot_float_dtype = mot_float_dtype

    @property
    def is_scalar(self):
        return True

    @property
    def read_data_back(self):
        return False

    def get_scalar_arg_dtype(self):
        if self._ctype.startswith('mot_float_type'):
            return self._mot_float_dtype
        return self._value.dtype

    def get_data(self):
        return self._value.astype(self.get_scalar_arg_dtype())

    def get_struct_declaration(self, name):
        return '{} {};'.format(self._ctype, name)

    def get_kernel_argument_declaration(self, name):
        return '{} {}'.format(self._ctype, name)

    def get_struct_init_string(self, name, problem_id_substitute):
        mot_dtype = SimpleCLDataType.from_string(self._ctype)
        if np.isinf(self._value):
            assignment = 'INFINITY'
        elif mot_dtype.is_vector_type:
            vector_length = mot_dtype.vector_length
            values = [str(val) for val in self._value[0]]
            if len(values) < vector_length:
                values.extend([str(0)] * (vector_length - len(values)))
            assignment = '(' + self._ctype + ')(' + ', '.join(values) + ')'
        else:
            assignment = str(self._value)
        return assignment

    def get_kernel_inputs(self, cl_context, workgroup_size):
        return self.get_data()

    def include_in_kernel_call(self):
        return False


class KernelLocalMemory(KernelData):

    def __init__(self, ctype, size_func=None):
        """Indicates that a local memory array of the indicated size must be loaded as kernel input data.

        Args:
            ctype (str): the desired c-type for this local memory object, like ``int``, ``float`` or ``mot_float_type``.
            size_func (Function): the function that can calculate the required local memory size (in number of bytes),
                given the workgroup size and numpy dtype in use by the kernel.
        """
        self._ctype = ctype
        self._size_func = size_func or (lambda workgroup_size, dtype: workgroup_size * np.dtype(dtype).itemsize)
        self._mot_float_dtype = None

    def set_mot_float_dtype(self, mot_float_dtype):
        self._mot_float_dtype = mot_float_dtype

    @property
    def is_scalar(self):
        return False

    @property
    def read_data_back(self):
        return False

    def get_scalar_arg_dtype(self):
        return None

    def get_data(self):
        return None

    def get_struct_declaration(self, name):
        return 'local {}* restrict {};'.format(self._ctype, name)

    def get_kernel_argument_declaration(self, name):
        return 'local {}* restrict {}'.format(self._ctype, name)

    def get_struct_init_string(self, name, problem_id_substitute):
        return name

    def get_kernel_inputs(self, cl_context, workgroup_size):
        if workgroup_size is None:
            raise ValueError("Can not initialize the local memory kernel data, the workgroup_size is None.")

        return cl.LocalMemory(self._size_func(
            workgroup_size, ctype_to_dtype(self._ctype, dtype_to_ctype(self._mot_float_dtype))))

    def include_in_kernel_call(self):
        return True


class KernelArray(KernelData):

    def __init__(self, data, ctype=None, offset_str=None, is_writable=False, is_readable=True, ensure_zero_copy=False):
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
            offset_str (str): the offset definition, can use ``{problem_id}`` for multiplication purposes. Set to 0
                for no offset.
            is_writable (boolean): if the data must be loaded writable or not, defaults to False
            is_readable (boolean): if this data must be made readable
            ensure_zero_copy (boolean): only used if ``is_writable`` is set to True. If set, we guarantee that the
                return values are written to the same input array. This allows the user of this class to user their
                reference to the underlying data, relieving the user of having to use :meth:`get_data`.
        """
        self._requirements = ['C', 'A', 'O']
        if is_writable:
            self._requirements.append('W')

        self._data = np.require(data, requirements=self._requirements)
        if ctype and not ctype.startswith('mot_float_type'):
            self._data = convert_data_to_dtype(self._data, ctype)

        if is_writable and ensure_zero_copy and self._data is not data:
            raise ValueError('Zero copy was set but we had to make '
                             'a copy to guarantee the "CAOW" requirements and the ctype requirements.')

        self._offset_str = offset_str
        self._is_writable = is_writable
        self._is_readable = is_readable
        self._ctype = ctype or dtype_to_ctype(data.dtype)
        self._mot_float_dtype = None
        self._backup_data_reference = None
        self._ensure_zero_copy = ensure_zero_copy

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

    def _get_offset_str(self, problem_id_substitute):
        if self._offset_str is None:
            offset_str = str(self._data.strides[0] // self._data.itemsize) + ' * {problem_id}'
        else:
            offset_str = str(self._offset_str)
        return offset_str.replace('{problem_id}', problem_id_substitute)

    @property
    def data_length(self):
        """Get the number of elements per problem instance.

        In the kernels, arrays are loaded as one dimensional arrays per problem instance. This property should return
        the length of that one dimensional array per problem instance.

        Returns:
            int: the number of elements per problem instance.
        """
        return self._data.strides[0] // self._data.itemsize

    @property
    def is_scalar(self):
        return False

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def read_data_back(self):
        return self._is_writable

    @property
    def is_readable(self):
        """If this kernel input data must be readable by the kernel.

        This is used in conjunction with :meth:`is_writable` when loading the data in the kernel.

        Returns:
            boolean: if this data must be made readable by the kernel function
        """
        return self._is_readable

    @property
    def is_writable(self):
        """Check if this kernel input data will write data back.

        This is used in conjunction with :meth:`is_readable` when loading the data in the kernel.

        If this returns true the kernel function must ensure that the data is loaded with at least write permissions.
        This flag will also ensure that the data will be read back from the device after kernel execution, overwriting
        the current data.

        Returns:
            boolean: if this data must be made writable and be read back after function execution.
        """
        return self._is_writable

    def get_struct_declaration(self, name):
        return 'global {}* restrict {};'.format(self._ctype, name)

    def get_kernel_argument_declaration(self, name):
        return 'global {}* restrict {}'.format(self._ctype, name)

    def get_struct_init_string(self, name, problem_id_substitute):
        return name + ' + ' + self._get_offset_str(problem_id_substitute)

    def get_scalar_arg_dtype(self):
        return None

    def get_data(self):
        return self._data

    def get_kernel_inputs(self, cl_context, workgroup_size):
        if self._is_writable:
            if self._is_readable:
                flags = cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR
            else:
                flags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR
        else:
            flags = cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR
        return cl.Buffer(cl_context, flags, hostbuf=self._data)

    def include_in_kernel_call(self):
        return True


class KernelAllocatedArray(KernelData):

    def __init__(self, shape, ctype, offset_str=None, is_writable=True, is_readable=False):
        """Allocate an output buffer of the given shape.

        This is similar to :class:`~mot.utils.KernelArray` although these objects are not readable but only
        writable by default.

        This is meant to quickly allocate a buffer large enough to hold the data requested. After running an OpenCL
        kernel you can get the written data using the method :meth:`get_data`.

        Args:
            shape (tuple): the shape of the output array
            offset_str (str): the offset definition, can use ``{problem_id}`` for multiplication purposes. Set to 0
                for no offset.
            is_writable (boolean): if the data must be loaded writable or not, defaults to True
            is_readable (boolean): if this data must be made readable, defaults to False
        """
        self._requirements = ['C', 'A', 'O']
        if is_writable:
            self._requirements.append('W')

        self._shape = shape
        self._data = None
        self._offset_str = offset_str
        self._is_writable = is_writable
        self._is_readable = is_readable
        self._ctype = ctype
        self._mot_float_dtype = None

    def set_mot_float_dtype(self, mot_float_dtype):
        dtype = ctype_to_dtype(self._ctype, mot_float_type=dtype_to_ctype(mot_float_dtype))
        self._mot_float_dtype = mot_float_dtype
        self._data = np.zeros(self._shape, dtype=dtype)
        self._data = np.require(self._data, requirements=self._requirements)

    def _get_offset_str(self, problem_id_substitute):
        if self._offset_str is None:
            offset_str = str(self._data.strides[0] // self._data.itemsize) + ' * {problem_id}'
        else:
            offset_str = str(self._offset_str)
        return offset_str.replace('{problem_id}', problem_id_substitute)

    @property
    def is_scalar(self):
        return False

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def read_data_back(self):
        return self._is_writable

    @property
    def is_readable(self):
        """If this kernel input data must be readable by the kernel.

        This is used in conjunction with :meth:`is_writable` when loading the data in the kernel.

        Returns:
            boolean: if this data must be made readable by the kernel function
        """
        return self._is_readable

    @property
    def is_writable(self):
        """Check if this kernel input data will write data back.

        This is used in conjunction with :meth:`is_readable` when loading the data in the kernel.

        If this returns true the kernel function must ensure that the data is loaded with at least write permissions.
        This flag will also ensure that the data will be read back from the device after kernel execution, overwriting
        the current data.

        Returns:
            boolean: if this data must be made writable and be read back after function execution.
        """
        return self._is_writable

    def get_struct_declaration(self, name):
        return 'global {}* restrict {};'.format(self._ctype, name)

    def get_kernel_argument_declaration(self, name):
        return 'global {}* restrict {}'.format(self._ctype, name)

    def get_struct_init_string(self, name, problem_id_substitute):
        return name + ' + ' + self._get_offset_str(problem_id_substitute)

    def get_scalar_arg_dtype(self):
        return None

    def get_data(self):
        return self._data

    def get_kernel_inputs(self, cl_context, workgroup_size):
        if self._is_writable:
            if self._is_readable:
                flags = cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR
            else:
                flags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR
        else:
            flags = cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR
        return cl.Buffer(cl_context, flags, hostbuf=self._data)

    def include_in_kernel_call(self):
        return True
