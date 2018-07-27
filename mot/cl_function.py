import collections

import numpy as np
from copy import copy

import tatsu

from mot.cl_data_type import SimpleCLDataType
from textwrap import dedent, indent

from mot.cl_routines.base import apply_cl_function
from mot.kernel_data import KernelData, KernelScalar, KernelArray
from mot.utils import is_scalar

__author__ = 'Robbert Harms'
__date__ = '2017-08-31'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


_simple_cl_function_parser = tatsu.compile('''
    result = [address_space] data_type function_name arglist body $;
    address_space = ['__'] ('local' | 'global' | 'constant' | 'private');
    data_type = /\w+(\s*(\*)?)+/;
    function_name = /\w+/;
    arglist = '(' @+:arg {',' @+:arg}* ')' | '()';
    arg = /[\w \*]+/;
    body = /\{(?s).*/;
''')


class CLFunction(object):
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
            list of :class:`mot.cl_function.CLFunctionParameter`: list of the parameters in this
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

    def get_cl_extra(self):
        """Get the extra CL code outside the function's body.

        Returns:
            str or None: the CL code outside the function body, can be None
        """
        raise NotImplementedError()

    def evaluate(self, inputs, nmr_instances=None, use_local_reduction=False, cl_runtime_info=None):
        """Evaluate this function for each set of given parameters.

        Given a set of input parameters, this model will be evaluated for every parameter set.
        This function will convert possible dots in the parameter names to underscores for use in the CL kernel.

        Args:
            inputs (dict[str: Union(ndarray, mot.utils.KernelData)]): for each parameter of the function
                the input data. Each of these input datasets must either be a scalar or be of equal length in the
                first dimension. The user can either input raw ndarrays or input KernelData objects.
                If an ndarray is given we will load it read/write by default.
            nmr_instances (int): the number of parallel processes to run. If not given, we try to autodetect it
                from the matrix with the largest number of rows (largest 1st dimension).
            use_local_reduction (boolean): set this to True if you want to use local memory reduction in
                 evaluating this function. If this is set to True we will multiply the global size
                 (given by the nmr_instances) by the work group sizes.
            cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the runtime information for execution

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


class SimpleCLFunction(CLFunction):

    def __init__(self, return_type, cl_function_name, parameter_list, cl_body, dependencies=(), cl_extra=None):
        """A simple implementation of a CL function.

        Args:
            return_type (str): the CL return type of the function
            cl_function_name (string): The name of the CL function
            parameter_list (list or tuple): This either contains instances of
                :class:`mot.cl_parameter.CLFunctionParameter` or contains tuples with arguments that
                can be used to construct a :class:`mot.cl_parameter.SimpleCLFunctionParameter`.
            cl_body (str): the body of the CL code for this function.
            dependencies (list or tuple of CLLibrary): The list of CL libraries this function depends on
            cl_extra (str): extra CL code for this function that does not warrant an own function.
                This is prepended to the function body.
        """
        super(SimpleCLFunction, self).__init__()
        self._return_type = return_type
        self._function_name = cl_function_name
        self._parameter_list = self._resolve_parameters(parameter_list)
        self._cl_body = cl_body
        self._dependencies = dependencies or []
        self._cl_extra = cl_extra

    @classmethod
    def from_string(cls, cl_function, dependencies=(), cl_extra=None):
        """Parse the given CL function into a SimpleCLFunction object.

        Args:
            cl_function (str): the function we wish to turn into an object
            dependencies (list or tuple of CLLibrary): The list of CL libraries this function depends on
            cl_extra (str): extra CL code for this function that does not warrant an own function.
                This is prepended to the function body.

        Returns:
            SimpleCLFunction: the CL data type for this parameter declaration
        """
        class Semantics(object):

            def __init__(self):
                self._return_type = ''
                self._function_name = ''
                self._parameter_list = []
                self._cl_body = ''

            def result(self, ast):
                return SimpleCLFunction(self._return_type, self._function_name, self._parameter_list, self._cl_body,
                                        dependencies=dependencies, cl_extra=cl_extra)

            def address_space(self, ast):
                self._return_type = ast.strip() + ' '
                return ast

            def data_type(self, ast):
                self._return_type += ''.join(ast).strip()
                return ast

            def function_name(self, ast):
                self._function_name = ast.strip()
                return ast

            def arglist(self, ast):
                if ast != '()':
                    self._parameter_list = ast
                return ast

            def body(self, ast):
                self._cl_body = ast.strip()[1:-1]
                return ast

        return _simple_cl_function_parser.parse(cl_function, semantics=Semantics())

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
            {cl_extra}
            {code}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=indent(self._get_cl_dependency_code(), ' ' * 4 * 3),
                   inclusion_guard_name='INCLUDE_GUARD_{}'.format(self.get_cl_function_name()),
                   cl_extra=self._cl_extra if self._cl_extra is not None else '',
                   code=indent('\n' + cl_code + '\n', ' ' * 4 * 3)))

    def get_cl_body(self):
        return self._cl_body

    def get_cl_extra(self):
        return self._cl_extra

    def evaluate(self, inputs, nmr_instances=None, use_local_reduction=False, cl_runtime_info=None):
        def wrap_input_data(input_data, nmr_instances):
            def get_kernel_data(param):
                if isinstance(input_data[param.name], KernelData):
                    return input_data[param.name]
                elif param.data_type.is_vector_type and np.squeeze(input_data[param.name]).shape[0] == 3:
                    return KernelScalar(input_data[param.name], ctype=param.data_type.ctype)
                elif is_scalar(input_data[param.name]) and not param.data_type.is_pointer_type:
                    return KernelScalar(input_data[param.name])
                else:
                    if is_scalar(input_data[param.name]):
                        data = np.ones(nmr_instances) * input_data[param.name]
                    else:
                        data = input_data[param.name]

                    return KernelArray(data, ctype=param.data_type.ctype, is_writable=True, is_readable=True)

            kernel_items = {}
            for param in self.get_parameters():
                if param.data_type.raw_data_type == 'mot_data_struct':
                    kernel_items.update(input_data[param.name])
                else:
                    kernel_items[param.name.replace('.', '_')] = get_kernel_data(param)

            return kernel_items

        def get_minimum_data_length(cl_function, input_data):
            min_length = 1

            for param in cl_function.get_parameters():
                value = input_data[param.name]

                if isinstance(value, collections.Mapping):
                    pass
                elif isinstance(input_data[param.name], KernelData):
                    data = value.get_data()
                    if data is not None:
                        if np.ndarray(data).shape[0] > min_length:
                            min_length = np.maximum(min_length, np.ndarray(data).shape[0])

                elif param.data_type.is_vector_type and np.squeeze(input_data[param.name]).shape[0] == 3:
                    pass
                elif is_scalar(input_data[param.name]):
                    pass
                else:
                    if isinstance(value, (tuple, list)):
                        min_length = np.maximum(min_length, len(value))
                    elif value.shape[0] > min_length:
                        min_length = np.maximum(min_length, value.shape[0])

            return min_length

        for param in self.get_parameters():
            if param.name not in inputs:
                names = [param.name for param in self.get_parameters()]
                missing_names = [name for name in names if name not in inputs]
                raise ValueError('Some parameters are missing an input value, '
                                 'required parameters are: {}, missing inputs are: {}'.format(names, missing_names))

        nmr_instances = nmr_instances or get_minimum_data_length(self, inputs)
        kernel_items = wrap_input_data(inputs, nmr_instances)

        return apply_cl_function(self, kernel_items, nmr_instances,
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


class CLFunctionParameter(object):

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
            mot.cl_data_type.SimpleCLDataType: The CL data type.
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
            data_type (mot.cl_data_type.SimpleCLDataType or str): the data type expected by this parameter
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
