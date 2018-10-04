import warnings
from collections import Iterable, Mapping
from collections.__init__ import OrderedDict

import numpy as np
from copy import copy

import pyopencl as cl

from mot.lib.cl_data_type import SimpleCLDataType
from textwrap import dedent, indent

from mot.configuration import CLRuntimeInfo
from mot.lib.kernel_data import KernelData, Scalar, Array, Zeros
from mot.lib.load_balance_strategies import Worker
from mot.lib.utils import is_scalar, get_float_type_def, split_cl_function

__author__ = 'Robbert Harms'
__date__ = '2017-08-31'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class CLCodeObject:
    """Interface for basic code objects."""

    def get_cl_code(self):
        """Get the CL code for this code object and all its dependencies, with include guards.

        Returns:
            str: The CL code for inclusion in a kernel.
        """
        raise NotImplementedError()


class CLFunction(CLCodeObject):
    """Interface for a basic CL function."""

    def get_return_type(self):
        """Get the type (in CL naming) of the returned value from this function.

        Returns:
            str: The return type of this CL function. (Examples: double, int, double4, ...)
        """
        raise NotImplementedError()

    def get_cl_function_name(self):
        """Return the calling name of the implemented CL function

        Returns:
            str: The name of this CL function
        """
        raise NotImplementedError()

    def get_parameters(self):
        """Return the list of parameters from this CL function.

        Returns:
            list of :class:`mot.lib.cl_function.CLFunctionParameter`: list of the parameters in this
                model in the same order as in the CL function"""
        raise NotImplementedError()

    def get_signature(self):
        """Get the CL signature of this function.

        Returns:
            str: the CL code for the signature of this CL function.
        """
        raise NotImplementedError()

    def get_cl_code(self):
        """Get the function code for this function and all its dependencies, with include guards.

        Returns:
            str: The CL code for inclusion in a kernel.
        """
        raise NotImplementedError()

    def get_cl_body(self):
        """Get the CL code for the body of this function.

        Returns:
            str: the CL code of this function body
        """
        raise NotImplementedError()

    def evaluate(self, inputs, nmr_instances, use_local_reduction=False, cl_runtime_info=None):
        """Evaluate this function for each set of given parameters.

        Given a set of input parameters, this model will be evaluated for every parameter set.
        This function will convert possible dots in the parameter names to underscores for use in the CL kernel.

        Args:
            inputs (Iterable[Union(ndarray, mot.lib.utils.KernelData)]
                    or Mapping[str: Union(ndarray, mot.lib.utils.KernelData)]): for each CL function parameter
                the input data. Each of these input datasets must either be a scalar or be of equal length in the
                first dimension. The elements can either be raw ndarrays or KernelData objects.
                If an ndarray is given we will load it read/write by default. You can provide either an iterable
                with one value per parameter, or a mapping with for every parameter a corresponding value.
            nmr_instances (int): the number of parallel processes to run.
            use_local_reduction (boolean): set this to True if you want to use local memory reduction in
                 evaluating this function. If this is set to True we will multiply the global size
                 (given by the nmr_instances) by the work group sizes.
            cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information for execution

        Returns:
            ndarray: the return values of the function, which can be None if this function has a void return type.
        """
        raise NotImplementedError()

    def get_dependencies(self):
        """Get the list of dependencies this function depends on.

        Returns:
            list[CLFunction]: the list of dependencies for this function.
        """
        raise NotImplementedError()


class SimpleCLCodeObject(CLCodeObject):

    def __init__(self, cl_code):
        """Simple code object for including type definitions in the kernel.

        Args:
            cl_code (str): CL code to be included in the kernel
        """
        self._cl_code = cl_code

    def get_cl_code(self):
        return self._cl_code


class SimpleCLFunction(CLFunction):

    def __init__(self, return_type, cl_function_name, parameter_list, cl_body, dependencies=None):
        """A simple implementation of a CL function.

        Args:
            return_type (str): the CL return type of the function
            cl_function_name (string): The name of the CL function
            parameter_list (list or tuple): This either contains instances of
                :class:`CLFunctionParameter` or contains tuples with arguments that
                can be used to construct a :class:`SimpleCLFunctionParameter`.
            cl_body (str): the body of the CL code for this function.
            dependencies (Iterable[CLCodeObject]): The CL code objects this function depends on,
                these will be prepended to the CL code generated by this function.
        """
        super().__init__()
        self._return_type = return_type
        self._function_name = cl_function_name
        self._parameter_list = self._resolve_parameters(parameter_list)
        self._cl_body = cl_body
        self._dependencies = dependencies or []

    @classmethod
    def from_string(cls, cl_function, dependencies=()):
        """Parse the given CL function into a SimpleCLFunction object.

        Args:
            cl_function (str): the function we wish to turn into an object
            dependencies (list or tuple of CLLibrary): The list of CL libraries this function depends on

        Returns:
            SimpleCLFunction: the CL data type for this parameter declaration
        """
        return_type, function_name, parameter_list, body = split_cl_function(cl_function)
        return SimpleCLFunction(return_type, function_name, parameter_list, body, dependencies=dependencies)

    def get_cl_function_name(self):
        return self._function_name

    def get_return_type(self):
        return self._return_type

    def get_parameters(self):
        return self._parameter_list

    def get_signature(self):
        return '{return_type} {cl_function_name}({parameters});'.format(
            return_type=self.get_return_type(),
            cl_function_name=self.get_cl_function_name(),
            parameters=', '.join(self._get_parameter_signatures()))

    def get_cl_code(self):
        cl_code = dedent('''
            {return_type} {cl_function_name}({parameters}){{
            {body}
            }}
        '''.format(return_type=self.get_return_type(),
                   cl_function_name=self.get_cl_function_name(),
                   parameters=', '.join(self._get_parameter_signatures()),
                   body=indent(dedent(self._cl_body), ' '*4*4)))

        return dedent('''
            {dependencies}
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            {code}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=indent(self._get_cl_dependency_code(), ' ' * 4 * 3),
                   inclusion_guard_name='INCLUDE_GUARD_{}'.format(self.get_cl_function_name()),
                   code=indent('\n' + cl_code + '\n', ' ' * 4 * 3)))

    def get_cl_body(self):
        return self._cl_body

    def evaluate(self, inputs, nmr_instances, use_local_reduction=False, cl_runtime_info=None):
        def wrap_input_data(input_data):
            def get_data_object(param):
                if input_data[param.name] is None:
                    return Scalar(0)
                elif isinstance(input_data[param.name], KernelData):
                    return input_data[param.name]
                elif param.data_type.is_vector_type and np.squeeze(input_data[param.name]).shape[0] == 3:
                    return Scalar(input_data[param.name], ctype=param.data_type.ctype)
                elif is_scalar(input_data[param.name]) \
                        and not (param.data_type.is_pointer_type or param.data_type.is_array_type):
                    return Scalar(input_data[param.name])
                else:
                    if is_scalar(input_data[param.name]):
                        data = np.ones(nmr_instances) * input_data[param.name]
                    else:
                        data = input_data[param.name]

                    if param.data_type.is_pointer_type or param.data_type.is_array_type:
                        return Array(data, ctype=param.data_type.ctype, mode='rw')
                    else:
                        return Array(data, ctype=param.data_type.ctype, mode='r', as_scalar=True)

            return {param.name.replace('.', '_'): get_data_object(param) for param in self.get_parameters()}

        if isinstance(inputs, Iterable) and not isinstance(inputs, Mapping):
            inputs = list(inputs)
            if len(inputs) != len(self.get_parameters()):
                raise ValueError('The length of the input list ({}), does not equal '
                                 'the number of parameters ({})'.format(len(inputs), len(self.get_parameters())))

            param_names = [param.name for param in self.get_parameters()]
            inputs = dict(zip(param_names, inputs))

        for param in self.get_parameters():
            if param.name not in inputs:
                names = [param.name for param in self.get_parameters()]
                missing_names = [name for name in names if name not in inputs]
                raise ValueError('Some parameters are missing an input value, '
                                 'required parameters are: {}, missing inputs are: {}'.format(names, missing_names))

        return apply_cl_function(self, wrap_input_data(inputs), nmr_instances,
                                 use_local_reduction=use_local_reduction, cl_runtime_info=cl_runtime_info)

    def get_dependencies(self):
        return self._dependencies

    def _get_parameter_signatures(self):
        """Get the signature of the parameters for the CL function declaration.

        This should return the list of signatures of the parameters for use inside the function signature.

        Returns:
            list: the signatures of the parameters for the use in the CL code.
        """
        return ['{} {}'.format(p.data_type.get_declaration(), p.name.replace('.', '_'))
                for p in self.get_parameters()]

    def _get_cl_dependency_code(self):
        """Get the CL code for all the CL code for all the dependencies.

        Returns:
            str: The CL code with the actual code.
        """
        code = ''
        for d in self._dependencies:
            code += d.get_cl_code() + "\n"
        return code

    @staticmethod
    def _resolve_parameters(parameter_list):
        params = []
        for param in parameter_list:
            if isinstance(param, CLFunctionParameter):
                params.append(param)
            elif isinstance(param, str):
                params.append(SimpleCLFunctionParameter.from_string(param))
            else:
                params.append(SimpleCLFunctionParameter(*param))
        return params

    def __str__(self):
        return dedent('''
            {return_type} {cl_function_name}({parameters}){{
            {body}
            }}
        '''.format(return_type=self.get_return_type(),
                   cl_function_name=self.get_cl_function_name(),
                   parameters=', '.join(self._get_parameter_signatures()),
                   body=indent(dedent(self._cl_body), ' '*4*4)))

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return type(self) != type(other)


class CLFunctionParameter:

    @property
    def name(self):
        """The name of this parameter.

        Returns:
            str: the name of this parameter
        """
        raise NotImplementedError()

    @property
    def data_type(self):
        """Get the CL data type of this parameter

        Returns:
            mot.lib.cl_data_type.SimpleCLDataType: The CL data type.
        """
        raise NotImplementedError()

    def get_renamed(self, name):
        """Get a copy of the current parameter but then with a new name.

        Args:
            name (str): the new name for this parameter

        Returns:
            cls: a copy of the current type but with a new name
        """
        raise NotImplementedError()


class SimpleCLFunctionParameter(CLFunctionParameter):

    def __init__(self, data_type, name):
        """Creates a new function parameter for the CL functions.

        Args:
            data_type (mot.lib.cl_data_type.SimpleCLDataType or str): the data type expected by this parameter
                If a string is given we will use ``SimpleCLDataType.from_string`` for translating the data_type.
            name (str): The name of this parameter

        Attributes:
            name (str): The name of this parameter
        """
        if isinstance(data_type, str):
            self._data_type = SimpleCLDataType.from_string(data_type)
        else:
            self._data_type = data_type

        self._name = name

    @classmethod
    def from_string(cls, parameter_string):
        """Generate a simple function parameter from a C string.

        This accepts for example items like ``int index`` and will parse from that the data type and parameter name.

        Args:
             parameter_string (str): the parameter string containing the data type and parameter name.

        Returns:
            SimpleCLFunctionParameter: an instantiated function parameter object
        """
        parameter_name = parameter_string.split(' ')[-1].strip()
        data_type = parameter_string[:-len(parameter_name)].strip()
        return SimpleCLFunctionParameter(data_type, parameter_name)

    @property
    def name(self):
        return self._name

    @property
    def data_type(self):
        return self._data_type

    def get_renamed(self, name):
        new_param = copy(self)
        new_param._name = name
        return new_param


def apply_cl_function(cl_function, kernel_data, nmr_instances, use_local_reduction=False, cl_runtime_info=None):
    """Run the given function/procedure on the given set of data.

    This class will wrap the given CL function in a kernel call and execute that that for every data instance using
    the provided kernel data. This class will respect the read write setting of the kernel data elements such that
    output can be written back to the according kernel data elements.

    Args:
        cl_function (mot.lib.cl_function.CLFunction): the function to
            run on the datasets. Either a name function tuple or an actual CLFunction object.
        kernel_data (dict[str: mot.lib.kernel_data.KernelData]): the data to use as input to the function.
        nmr_instances (int): the number of parallel threads to run (used as ``global_size``)
        use_local_reduction (boolean): set this to True if you want to use local memory reduction in
             your CL procedure. If this is set to True we will multiply the global size (given by the nmr_instances)
             by the work group sizes.
        cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information
    """
    cl_runtime_info = cl_runtime_info or CLRuntimeInfo()

    for param in cl_function.get_parameters():
        if param.name not in kernel_data:
            names = [param.name for param in cl_function.get_parameters()]
            missing_names = [name for name in names if name not in kernel_data]
            raise ValueError('Some parameters are missing an input value, '
                             'required parameters are: {}, missing inputs are: {}'.format(names, missing_names))

    if cl_function.get_return_type() != 'void':
        kernel_data['_results'] = Zeros((nmr_instances,), cl_function.get_return_type())

    workers = []
    for cl_environment in cl_runtime_info.get_cl_environments():
        workers.append(_ProcedureWorker(cl_environment, cl_runtime_info.get_compile_flags(),
                                        cl_function,
                                        kernel_data, cl_runtime_info.double_precision, use_local_reduction))

    cl_runtime_info.load_balancer.process(workers, nmr_instances)

    if cl_function.get_return_type() != 'void':
        return kernel_data['_results'].get_data()


class _ProcedureWorker(Worker):

    def __init__(self, cl_environment, compile_flags, cl_function,
                 kernel_data, double_precision, use_local_reduction):
        super().__init__(cl_environment)
        self._cl_function = cl_function
        self._kernel_data = OrderedDict(sorted(kernel_data.items()))
        self._double_precision = double_precision
        self._use_local_reduction = use_local_reduction

        self._mot_float_dtype = np.float32
        if double_precision:
            self._mot_float_dtype = np.float64

        for data in self._kernel_data.values():
            data.set_mot_float_dtype(self._mot_float_dtype)

        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

        self._workgroup_size = self._kernel.run_procedure.get_work_group_info(
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            self._cl_environment.device)
        if not self._use_local_reduction:
            self._workgroup_size = 1

        self._kernel_inputs = {name: data.get_kernel_inputs(self._cl_context, self._workgroup_size)
                               for name, data in self._kernel_data.items()}

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        func = self._kernel.run_procedure
        func.set_scalar_arg_dtypes(self.get_scalar_arg_dtypes())

        kernel_inputs_list = []
        for inputs in [self._kernel_inputs[name] for name in self._kernel_data]:
            kernel_inputs_list.extend(inputs)

        func(self._cl_queue,
             (int(nmr_problems * self._workgroup_size),),
             (int(self._workgroup_size),),
             *kernel_inputs_list,
             global_offset=(int(range_start * self._workgroup_size),))

        for name, data in self._kernel_data.items():
            data.enqueue_readouts(self._cl_queue, self._kernel_inputs[name], range_start, range_end)

    def _build_kernel(self, kernel_source, compile_flags=()):
        """Convenience function for building the kernel for this worker.

        Args:
            kernel_source (str): the kernel source to use for building the kernel

        Returns:
            cl.Program: a compiled CL kernel
        """
        from mot import configuration
        if configuration.should_ignore_kernel_compile_warnings():
            warnings.simplefilter("ignore")
        return cl.Program(self._cl_context, kernel_source).build(' '.join(compile_flags))

    def _get_kernel_source(self):
        assignment = ''
        if self._cl_function.get_return_type() != 'void':
            assignment = '__results[gid] = '

        variable_inits = []
        function_call_inputs = []
        post_function_callbacks = []
        for parameter in self._cl_function.get_parameters():
            data = self._kernel_data[parameter.name]
            call_args = (parameter.name, '_' + parameter.name, 'gid', parameter.data_type.address_space)

            variable_inits.append(data.initialize_variable(*call_args))
            function_call_inputs.append(data.get_function_call_input(*call_args))
            post_function_callbacks.append(data.post_function_callback(*call_args))

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += '\n'.join(data.get_type_definitions() for data in self._kernel_data.values())
        kernel_source += self._cl_function.get_cl_code()
        kernel_source += '''
            __kernel void run_procedure(''' + ",\n".join(self._get_kernel_arguments()) + '''){
                ulong gid = (ulong)(get_global_id(0) / get_local_size(0));
                
                ''' + '\n'.join(variable_inits) + '''     
                
                ''' + assignment + ' ' + self._cl_function.get_cl_function_name() + '(' + \
                         ', '.join(function_call_inputs) + ''');
                
                ''' + '\n'.join(post_function_callbacks) + '''
            }
        '''
        return kernel_source

    def _get_kernel_arguments(self):
        """Get the list of kernel arguments for loading the kernel data elements into the kernel.

        This will use the sorted keys for looping through the kernel input items.

        Returns:
            list of str: the list of parameter definitions
        """
        declarations = []
        for name, data in self._kernel_data.items():
            declarations.extend(data.get_kernel_parameters('_' + name))
        return declarations

    def get_scalar_arg_dtypes(self):
        """Get the location and types of the input scalars.

        Returns:
            list: for every kernel input element either None if the data is a buffer or the numpy data type if
                if is a scalar.
        """
        dtypes = []
        for name, data in self._kernel_data.items():
            dtypes.extend(data.get_scalar_arg_dtypes())
        return dtypes
