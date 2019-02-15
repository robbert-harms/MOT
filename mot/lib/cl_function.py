from collections import Iterable, Mapping
from collections.__init__ import OrderedDict

import numpy as np
from copy import copy

import pyopencl as cl
import tatsu

from textwrap import dedent, indent

from mot.configuration import CLRuntimeInfo
from mot.lib.kernel_data import KernelData, Scalar, Array, Zeros
from mot.lib.utils import is_scalar, get_float_type_def, split_cl_function, split_in_batches, get_atomic_functions

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
                :class:`CLFunctionParameter` or strings from which to form the function parameters.
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
                elif param.is_vector_type and np.squeeze(input_data[param.name]).shape[0] == 3:
                    return Scalar(input_data[param.name], ctype=param.ctype)
                elif is_scalar(input_data[param.name]) \
                        and not (param.is_pointer_type or param.is_array_type):
                    return Scalar(input_data[param.name])
                else:
                    if is_scalar(input_data[param.name]):
                        data = np.ones(nmr_instances) * input_data[param.name]
                    else:
                        data = input_data[param.name]

                    if param.is_pointer_type or param.is_array_type:
                        return Array(data, ctype=param.ctype, mode='rw')
                    else:
                        return Array(data, ctype=param.ctype, mode='r', as_scalar=True)

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
        declarations = []
        for p in self.get_parameters():
            new_p = p.get_renamed(p.name.replace('.', '_'))
            declarations.append(new_p.get_declaration())
        return declarations

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
            else:
                params.append(SimpleCLFunctionParameter(param))
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

    def get_declaration(self):
        """Get the complete CL declaration for this parameter.

        Returns:
            str: the declaration for this data type.
        """
        raise NotImplementedError()

    @property
    def ctype(self):
        """Get the ctype of this data type.

        For example, if the data type is float4*, we will return float4 here.

        Returns:
            str: the full ctype of this data type
        """
        raise NotImplementedError()

    @property
    def address_space(self):
        """Get the address space of this data declaration.

        Returns:
            str: the data type address space, one of ``global``, ``local``, ``constant`` or ``private``.
        """
        raise NotImplementedError()

    @property
    def basic_ctype(self):
        """Get the basic data type without the vector and pointer additions.

        For example, if the full data ctype is ``float4*``, we will only return ``float`` here.

        Returns:
            str: the raw CL data type
        """
        raise NotImplementedError()

    @property
    def is_vector_type(self):
        """Check if this data type is a vector type (like for example double4, float2, int8, etc.).

        Returns:
            boolean: True if it is a vector type, false otherwise
        """
        raise NotImplementedError()

    @property
    def vector_length(self):
        """Get the length of this vector, returns None if not a vector type.

        Returns:
            int: the length of the vector type (for example, if the data type is float4, this returns 4).
        """
        raise NotImplementedError()

    @property
    def is_pointer_type(self):
        """Check if this parameter is a pointer type (appended by a ``*``)

        Returns:
            boolean: True if it is a pointer type, false otherwise
        """
        raise NotImplementedError()

    @property
    def nmr_pointers(self):
        """Get the number of asterisks / pointer references of this data type.

        If the data type is float**, we return 2 here.

        Returns:
            int: the number of pointer asterisks in the data type.
        """
        raise NotImplementedError()

    @property
    def array_sizes(self):
        """Get the dimension of this array type.

        This returns for example (10, 5) for the data type float[10][5].

        Returns:
            Tuple[int]: the sizes of the arrays
        """
        raise NotImplementedError()

    @property
    def is_array_type(self):
        """Check if this parameter is an array type (like float[3] or int[10][5]).

        Returns:
            boolean: True if this is an array type, false otherwise
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


_cl_data_type_parser = tatsu.compile('''
    result = [address_space] {type_qualifiers}* ctype {pointer_star}* {pointer_qualifiers}* name {array_size}*;

    address_space = ['__'] ('local' | 'global' | 'constant' | 'private');
    type_qualifiers = 'const' | 'volatile';

    basic_ctype = ?'(unsigned )?\w[\w]*[a-zA-Z]'; 
    vector_type_length = '2' | '3' | '4' | '8' | '16';
    ctype = basic_ctype [vector_type_length];
    pointer_star = '*';
    
    pointer_qualifiers = 'const' | 'restrict';

    name = /[\w\_\-\.]+/;
    array_size = /\[\d+\]/;
''')


class SimpleCLFunctionParameter(CLFunctionParameter):

    def __init__(self, declaration):
        """Creates a new function parameter for the CL functions.

        Args:
            declaration (str): the declaration of this parameter. For example ``global int foo``.
        """
        self._address_space = None
        self._type_qualifiers = []
        self._basic_ctype = ''
        self._vector_type_length = None
        self._nmr_pointer_stars = 0
        self._pointer_qualifiers = []
        self._name = ''
        self._array_sizes = []

        param = self

        class Semantics:

            def type_qualifiers(self, ast):
                if ast in param._type_qualifiers:
                    raise ValueError('The pre-type qualifier "{}" is present multiple times.'.format(ast))
                param._type_qualifiers.append(ast)
                return ast

            def address_space(self, ast):
                param._address_space = ''.join(ast)
                return ''.join(ast)

            def basic_ctype(self, ast):
                param._basic_ctype = ast
                return ast

            def vector_type_length(self, ast):
                param._vector_type_length = int(ast)
                return ast

            def pointer_star(self, ast):
                param._nmr_pointer_stars += 1
                return ast

            def pointer_qualifiers(self, ast):
                if ast in param._pointer_qualifiers:
                    raise ValueError('The pre-type qualifier "{}" is present multiple times.'.format(ast))
                param._pointer_qualifiers.append(ast)
                return ast

            def name(self, ast):
                param._name = ast
                return ast

            def array_size(self, ast):
                param._array_sizes.append(int(ast[1:-1]))
                return ast

        _cl_data_type_parser.parse(declaration, semantics=Semantics())

    @property
    def name(self):
        return self._name

    def get_renamed(self, name):
        new_param = copy(self)
        new_param._name = name
        return new_param

    def get_declaration(self):
        declaration = ''

        if self._address_space:
            declaration += str(self._address_space) + ' '

        if self._type_qualifiers:
            declaration += str(' '.join(self._type_qualifiers)) + ' '

        declaration += str(self.ctype)
        declaration += '*' * self._nmr_pointer_stars

        if self._pointer_qualifiers:
            declaration += ' ' + str(' '.join(self._pointer_qualifiers)) + ' '

        declaration += ' ' + self._name

        for s in self._array_sizes:
            declaration += '[{}]'.format(s)

        return declaration

    @property
    def ctype(self):
        if self._vector_type_length is not None:
            return '{}{}'.format(self._basic_ctype, self._vector_type_length)
        return self._basic_ctype

    @property
    def address_space(self):
        return self._address_space or 'private'

    @property
    def basic_ctype(self):
        return self._basic_ctype

    @property
    def is_vector_type(self):
        return self._vector_type_length is not None

    @property
    def vector_length(self):
        return self._vector_type_length

    @property
    def is_pointer_type(self):
        return self._nmr_pointer_stars > 0

    @property
    def nmr_pointers(self):
        return self._nmr_pointer_stars

    @property
    def array_sizes(self):
        return self._array_sizes

    @property
    def is_array_type(self):
        return len(self.array_sizes) > 0


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
    cl_environments = cl_runtime_info.cl_environments

    for param in cl_function.get_parameters():
        if param.name not in kernel_data:
            names = [param.name for param in cl_function.get_parameters()]
            missing_names = [name for name in names if name not in kernel_data]
            raise ValueError('Some parameters are missing an input value, '
                             'required parameters are: {}, missing inputs are: {}'.format(names, missing_names))

    if cl_function.get_return_type() != 'void':
        kernel_data['_results'] = Zeros((nmr_instances,), cl_function.get_return_type())

    workers = []
    for ind, cl_environment in enumerate(cl_environments):
        worker = _ProcedureWorker(cl_environment, cl_runtime_info.compile_flags,
                                  cl_function, kernel_data, cl_runtime_info.double_precision, use_local_reduction)
        workers.append(worker)

    def enqueue_batch(batch_size, offset):
        items_per_worker = [batch_size // len(cl_environments) for _ in range(len(cl_environments) - 1)]
        items_per_worker.append(batch_size - sum(items_per_worker))

        for ind, worker in enumerate(workers):
            worker.calculate(offset, offset + items_per_worker[ind])
            offset += items_per_worker[ind]
            worker.cl_queue.flush()

        for worker in workers:
            worker.cl_queue.finish()

        return offset

    total_offset = 0
    for batch_start, batch_end in split_in_batches(nmr_instances, 1e4 * len(workers)):
        total_offset = enqueue_batch(batch_end - batch_start, total_offset)

    if cl_function.get_return_type() != 'void':
        return kernel_data['_results'].get_data()


class _ProcedureWorker:

    def __init__(self, cl_environment, compile_flags, cl_function,
                 kernel_data, double_precision, use_local_reduction):

        self._cl_environment = cl_environment
        self._cl_context = cl_environment.context
        self._cl_queue = cl_environment.queue
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

    @property
    def cl_environment(self):
        """Get the used CL environment.

        Returns:
            cl_environment (CLEnvironment): The cl environment to use for calculations.
        """
        return self._cl_environment

    @property
    def cl_queue(self):
        """Get the queue this worker is using for its GPU computations.

        The load balancing routine will use this queue to flush and finish the computations.

        Returns:
            pyopencl queues: the queue used by this worker
        """
        return self._cl_queue

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
            call_args = (parameter.name, '_' + parameter.name, 'gid', parameter.address_space)

            variable_inits.append(data.initialize_variable(*call_args))
            function_call_inputs.append(data.get_function_call_input(*call_args))
            post_function_callbacks.append(data.post_function_callback(*call_args))

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += get_atomic_functions(self._double_precision)
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
