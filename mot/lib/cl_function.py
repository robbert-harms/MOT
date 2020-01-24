import os
from pkg_resources import resource_filename

from collections import Iterable
from collections.__init__ import OrderedDict
import numpy as np
from copy import copy
import pyopencl as cl
import tatsu
from textwrap import dedent, indent
from mot.configuration import CLRuntimeInfo
from mot.lib.kernel_data import Zeros, Array
from mot.lib.utils import get_cl_utility_definitions, split_cl_function, convert_inputs_to_kernel_data

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

    def is_kernel_func(self):
        """Check if this function is a kernel function or not.

        Kernel functions have the keyword 'kernel' in them, like: ``kernel void foo();``.

        Returns:
            boolean: if this function is a kernel function or not
        """
        raise NotImplementedError()

    def created_wrapped_kernel_func(self, input_data, kernel_name=None, context_variables=None):
        """Wrap the current CLFunction with a kernel CLFunction.

        The idea is that we can have a function like:

        .. code-block:: c

            int foo(void* data){...}

        That is, without the ``kernel`` modifier. This function can not readily be executed on a CL device.
        To make life easy, this method wraps the current CL code in a kernel like this:

        .. code-block:: c

            int foo(void* data){...}
            kernel void kernel_foo(...){
                ... data = ...;
                foo(&data);
            }

        And then kernel_foo can be executed on the device.

        In order to generate the correct kernel arguments, this method needs to know which data will be loaded and
        which with signature. At the moment, this is done by providing it with all the kernel inputs you wish to load
        into the kernel at execution time. The generated kernel function will then do do automatic data marshalling
        from all the :class:`mot.lib.kernel_data.KernelData` inputs to the inputs for the wrapped function.

        Args:
            input_data (Dict[str: mot.lib.kernel_data.KernelData]): mapping parameter names to kernel data objects.
            kernel_name (str): the name of the generated kernel function. If not given it will be called
                ``kernel_<CLFunction.get_cl_function_name()>``.
            context_variables (Dict[str: mot.lib.kernel_data.KernelData]): context variables we also need to load

        Returns:
            CLFunction: A CL function with :meth:`is_kernel_func` set to True.
        """
        raise NotImplementedError()

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

    def evaluate(self, inputs, nmr_instances, context_variables=None, enable_rng=False,
                 use_local_reduction=False, local_size=None, cl_runtime_info=None):
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
            context_variables (dict[str: mot.lib.kernel_data.KernelData]): data structures that will be loaded
                as program scope global variables. Note that not all KernelData types are allowed, only the
                global variables are allowed.
            enable_rng (boolean): if this function wants to use random numbers. If set to true we prepare the random
                number generator for use in this function.
            use_local_reduction (boolean): set this to True if you want to use local memory reduction in
                 evaluating this function. If this is set to True we will multiply the global size
                 (given by the nmr_instances) by the work group sizes.
            local_size (int): can be used to specify the exact local size (workgroup size) the kernel must use.
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

    def __init__(self, return_type, cl_function_name, parameter_list, cl_body, dependencies=None, is_kernel_func=False):
        """A simple implementation of a CL function.

        Args:
            return_type (str): the CL return type of the function
            cl_function_name (string): The name of the CL function
            parameter_list (list or tuple): This either contains instances of
                :class:`CLFunctionParameter` or strings from which to form the function parameters.
            cl_body (str): the body of the CL code for this function.
            dependencies (Iterable[CLCodeObject]): The CL code objects this function depends on,
                these will be prepended to the CL code generated by this function.
            is_kernel_func (boolean): if this function should be a kernel function
        """
        super().__init__()
        self._return_type = return_type
        self._function_name = cl_function_name
        self._parameter_list = self._resolve_parameters(parameter_list)
        self._cl_body = cl_body
        self._dependencies = dependencies or []
        self._is_kernel_func = is_kernel_func

    @classmethod
    def from_string(cls, cl_function, dependencies=()):
        """Parse the given CL function into a SimpleCLFunction object.

        Args:
            cl_function (str): the function we wish to turn into an object
            dependencies (list or tuple of CLLibrary): The list of CL libraries this function depends on

        Returns:
            SimpleCLFunction: the CL data type for this parameter declaration
        """
        is_kernel_func, return_type, function_name, parameter_list, body = split_cl_function(cl_function)
        return SimpleCLFunction(return_type, function_name, parameter_list, body,
                                dependencies=dependencies, is_kernel_func=is_kernel_func)

    def get_cl_function_name(self):
        return self._function_name

    def is_kernel_func(self):
        return self._is_kernel_func

    def get_return_type(self):
        return self._return_type

    def get_parameters(self):
        return self._parameter_list

    def created_wrapped_kernel_func(self, input_data, kernel_name=None, context_variables=None):
        kernel_name = kernel_name or 'kernel_' + self.get_cl_function_name()

        assignment = ''
        if self.get_return_type() != 'void':
            assignment = '__results[gid] = '

        context_variables = context_variables or {}
        context_inits = []
        for name, data in context_variables.items():
            context_inits.append(data.get_context_variable_initialization(name, '_context_' + name))

        variable_inits = []
        function_call_inputs = []
        for parameter in self.get_parameters():
            data = input_data[parameter.name]
            call_args = (parameter.name, '_' + parameter.name, 'gid')

            variable_inits.append(data.initialize_variable(*call_args))
            function_call_inputs.append(data.get_function_call_input(*call_args))

        parameter_list = []
        for name, data in input_data.items():
            parameter_list.extend(data.get_kernel_parameters('_' + name))

        for name, data in context_variables.items():
            parameter_list.extend(data.get_kernel_parameters('_context_' + name))

        cl_body = '''
            ulong gid = (ulong)(get_global_id(0) / get_local_size(0));

            ''' + '\n'.join(variable_inits) + '''
            ''' + '\n'.join(context_inits) + '''

            ''' + assignment + ' ' + self.get_cl_function_name() + '(' + ', '.join(function_call_inputs) + ''');
        '''
        return SimpleCLFunction('void', kernel_name, parameter_list, cl_body,
                                dependencies=[self], is_kernel_func=True)

    def get_signature(self):
        return dedent('{kernel} {return_type} {cl_function_name}({parameters});'.format(
            kernel='kernel' if self.is_kernel_func() else '',
            return_type=self.get_return_type(),
            cl_function_name=self.get_cl_function_name(),
            parameters=', '.join(self._get_parameter_signatures())))

    def get_cl_code(self):
        cl_code = dedent('''
            {kernel} {return_type} {cl_function_name}({parameters}){{
            {body}
            }}
        '''.format(kernel='kernel' if self.is_kernel_func() else '',
                   return_type=self.get_return_type(),
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

    def evaluate(self, inputs, nmr_instances, use_local_reduction=False, local_size=None,
                 context_variables=None, enable_rng=False, cl_runtime_info=None):
        processor = Processor(self, inputs, nmr_instances, use_local_reduction=use_local_reduction,
                              local_size=local_size, context_variables=context_variables, enable_rng=enable_rng,
                              cl_runtime_info=cl_runtime_info)
        processor.enqueue_run()
        processor.enqueue_finish()
        return processor.get_function_results()

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

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


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


class Processor:

    def __init__(self, cl_function, inputs, nmr_instances, use_local_reduction=False, local_size=None,
                 context_variables=None, enable_rng=False, cl_runtime_info=None):
        """Create a processor for the given function and inputs.

        By changing the data underlying the kernel data elements, this processor allows for multiple executions
        with only a low overhead.

        The typical way of using this processor is by:
        1) create it
        2) use :meth:`get_kernel_data` to get the kernel data elements and use get_data() for each of them to get the
            underlying data. You can then modify that.
        3) call :meth:`enqueue_run` to run this function on all devices with the current data
        4) call :meth:`enqueue_finish` to finish the execution (or do not call it and chain another operation)
        5) optionally use :meth:`get_function_results` to get the function results (if the function had a non-void
            return signature).

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
            local_size (int): can be used to specify the exact local size (workgroup size) the kernel must use.
            context_variables (dict[str: mot.lib.kernel_data.KernelData]): data structures that will be loaded
                as program scope global variables. Note that not all KernelData types are allowed, only the
                global variables are allowed.
            enable_rng (boolean): if this function wants to use random numbers. If set to true we prepare the random
                number generator for use in this function.
            cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information for execution
        """
        self._cl_function = cl_function
        self._kernel_data = convert_inputs_to_kernel_data(inputs, cl_function.get_parameters(), nmr_instances)

        context_variables = context_variables or {}
        if enable_rng:
            rng_state = np.random.uniform(low=np.iinfo(np.uint32).min, high=np.iinfo(np.uint32).max + 1,
                                          size=(nmr_instances, 8)).astype(np.uint32)
            context_variables['__rng_state'] = Array(rng_state, 'uint', mode='rw', ensure_zero_copy=True)

        cl_runtime_info = cl_runtime_info or CLRuntimeInfo()
        cl_environments = cl_runtime_info.cl_environments

        if cl_function.get_return_type() != 'void':
            self._kernel_data['_results'] = Zeros((nmr_instances,), cl_function.get_return_type())

        mot_float_dtype = np.float32
        if cl_runtime_info.double_precision:
            mot_float_dtype = np.float64

        for data in self._kernel_data.values():
            data.set_mot_float_dtype(mot_float_dtype)

        for data in context_variables.values():
            data.set_mot_float_dtype(mot_float_dtype)

        self._workers = []
        for ind, cl_environment in enumerate(cl_environments):
            worker = KernelWorker(cl_function, self._kernel_data, cl_environment,
                                  compile_flags=cl_runtime_info.compile_flags,
                                  double_precision=cl_runtime_info.double_precision,
                                  use_local_reduction=use_local_reduction,
                                  local_size=local_size,
                                  enable_rng=enable_rng, context_variables=context_variables)
            self._workers.append(worker)
        self._batches = cl_runtime_info.load_balancer.get_division(cl_environments, nmr_instances)

    def enqueue_run(self, flush=True, finish=False):
        """Enqueues and flushes running this function on all devices.

        Args:
            flush (boolean): if we flush the queues after enqueuing the work
            finish (boolean): if enqueue a finish operation to the queues after enqueueing the work
        """
        for worker, (batch_start, batch_end) in zip(self._workers, self._batches):
            if batch_end - batch_start > 0:
                worker.calculate(batch_start, batch_end)

                if flush:
                    worker.cl_queue.flush()
        if finish:
            self.enqueue_finish()

    def enqueue_finish(self):
        """Enqueues a finish operation to all the queues."""
        for worker in self._workers:
            worker.cl_queue.finish()

    def get_kernel_data(self):
        """Get the kernel data object this processor is working on"""
        return self._kernel_data

    def get_function_results(self):
        """Get the current function results. Only useful if the function has a non-void return type.

        Returns:
            ndarray: the return values of the function, which can be None if this function has a void return type.
        """
        if self._cl_function.get_return_type() != 'void':
            return self._kernel_data['_results'].get_data()


class KernelWorker:

    def __init__(self, cl_function, kernel_data, cl_environment, compile_flags=None,
                 double_precision=False, use_local_reduction=False, local_size=None,
                 context_variables=None, enable_rng=False):
        """Create a processor able to process the given function with the given data in the given environment.

        Objects of this type can be used in pipelines since very fast execution can be achieved by creating it once
        and then changing the underlying data of the kernel data objects.
        """
        context_variables = context_variables or {}

        self._cl_environment = cl_environment
        self._cl_context = cl_environment.context
        self._cl_queue = cl_environment.queue
        self._kernel_data = OrderedDict(sorted(kernel_data.items()))
        self._context_variables = OrderedDict(sorted(context_variables.items()))
        self._double_precision = double_precision
        self._use_local_reduction = use_local_reduction
        self._enable_rng = enable_rng

        self._scalar_arg_dtypes = []
        self._kernel_arguments = []

        for name, data in self._kernel_data.items():
            self._scalar_arg_dtypes.extend(data.get_scalar_arg_dtypes())
            self._kernel_arguments.extend(data.get_kernel_parameters('_' + name))

        for name, data in self._context_variables.items():
            self._scalar_arg_dtypes.extend(data.get_scalar_arg_dtypes())
            self._kernel_arguments.extend(data.get_kernel_parameters('_context_' + name))

        if cl_function.is_kernel_func():
            self._cl_function_kernel = cl_function
        else:
            self._cl_function_kernel = cl_function.created_wrapped_kernel_func(
                self._kernel_data, context_variables=context_variables)

        self._kernel = cl.Program(self._cl_context, self._get_kernel_source()).build(' '.join(compile_flags))
        self._kernel_func = getattr(self._kernel, self._cl_function_kernel.get_cl_function_name())

        if self._use_local_reduction:
            if local_size:
                self._workgroup_size = local_size
            else:
                self._workgroup_size = self._kernel_func.get_work_group_info(
                    cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                    self._cl_environment.device)
        else:
            self._workgroup_size = 1

        self._kernel_inputs = {}
        self._kernel_inputs_order = []

        for name, data in self._kernel_data.items():
            self._kernel_inputs[name] = data.get_kernel_inputs(self._cl_context, self._workgroup_size)
            self._kernel_inputs_order.append(name)

        for name, data in self._context_variables.items():
            self._kernel_inputs['_context_' + name] = data.get_kernel_inputs(self._cl_context, self._workgroup_size)
            self._kernel_inputs_order.append('_context_' + name)

        self._kernel_func.set_scalar_arg_dtypes(self._scalar_arg_dtypes)

    @property
    def cl_queue(self):
        """Get the queue this worker is using for its GPU computations.

        This may be used to flush or finish the queue to provide synchronization.

        Returns:
            pyopencl queue: the queue used by this worker
        """
        return self._cl_queue

    def calculate(self, range_start, range_end):
        """Start processing the current data on the given range.

        Args:
            range_start (int): the beginning of the range we will process (defines the start of the global offset)
            range_end (int): the end of the range we will process
        """
        nmr_problems = range_end - range_start

        kernel_inputs_list = []
        for inputs in [self._kernel_inputs[name] for name in self._kernel_inputs_order]:
            kernel_inputs_list.extend(inputs)

        for name, data in self._kernel_data.items():
            data.enqueue_device_access(self._cl_queue, self._kernel_inputs[name], range_start, range_end)

        self._kernel_func(
            self._cl_queue,
            (int(nmr_problems * self._workgroup_size),),
            (int(self._workgroup_size),),
            *kernel_inputs_list,
            global_offset=(int(range_start * self._workgroup_size),))

        for name, data in self._kernel_data.items():
            data.enqueue_host_access(self._cl_queue, self._kernel_inputs[name], range_start, range_end)

    def _get_kernel_source(self):
        kernel_source = ''
        kernel_source += get_cl_utility_definitions(self._double_precision)
        kernel_source += '\n'.join(data.get_type_definitions() for data in self._kernel_data.values())
        kernel_source += '\n'.join(data.get_type_definitions() for data in self._context_variables.values())
        kernel_source += '\n'.join(data.get_context_variable_declaration(name)
                                   for name, data in self._context_variables.items())
        if self._enable_rng:
            kernel_source += self._get_rng_cl_code()
        kernel_source += self._cl_function_kernel.get_cl_code()
        return kernel_source

    def _get_rng_cl_code(self):
        generator = 'philox'

        src = open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/openclfeatures.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/array.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/{}.h'.format(generator)), ),
                    'r').read()
        src += (open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/rand123.h'), ), 'r').read() % {
            'GENERATOR_NAME': generator})

        return src
