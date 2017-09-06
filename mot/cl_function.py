from textwrap import dedent, indent

from mot.cl_routines.mapping.cl_function_evaluator import CLFunctionEvaluator

__author__ = 'Robbert Harms'
__date__ = '2017-08-31'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class CLHeader(object):
    """Signature for a basic CL function."""

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
            list of CLFunctionParameter: list of the parameters in this model in the same order as in the CL function"""
        raise NotImplementedError()


class CLFunction(CLHeader):
    """Interface for a basic CL function."""

    def get_cl_code(self):
        """Get the function code for this function and all its dependencies, with include guards.

        Returns:
            str: The CL code for inclusion in a kernel.
        """
        raise NotImplementedError()

    def get_raw_cl_code(self):
        """Get the CL code of only the implementing function, without include guards.

        Returns:
            str: the CL for encapsulation
        """
        raise NotImplementedError()

    def evaluate(self, inputs, double_precision=False):
        """Evaluate this function for each set of given parameters.

        Given a set of input parameters, this model will be evaluated for every parameter set.

        Args:
            inputs (dict): for each parameter of the function an array with input data.
                Each of these input arrays must be of equal length in the first dimension.
            double_precision (boolean): if the function should be evaluated in double precision or not

        Returns:
            ndarray: a single array of the specified return type with for each parameter tuple an evaluation result
        """
        raise NotImplementedError()


class SimpleCLHeader(CLHeader):

    def __init__(self, return_type, cl_function_name, parameter_list):
        """A simple implementation of a CL header.

        Args:
            return_type (str): the CL return type of the function
            cl_function_name (string): The name of the CL function
            parameter_list (list or tuple of CLFunctionParameter): The list of parameters required for this function
        """
        super(SimpleCLHeader, self).__init__()
        self._return_type = return_type
        self._function_name = cl_function_name
        self._parameter_list = parameter_list

    def get_cl_function_name(self):
        return self._function_name

    def get_return_type(self):
        return self._return_type

    def get_parameters(self):
        return self._parameter_list

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return type(self) != type(other)


class SimpleCLFunction(CLFunction):

    def __init__(self, return_type, cl_function_name, parameter_list, cl_code, dependency_list=()):
        """A simple implementation of a CL function.

        Args:
            return_type (str): the CL return type of the function
            cl_function_name (string): The name of the CL function
            parameter_list (list or tuple of CLFunctionParameter): The list of parameters required for this function
            dependency_list (list or tuple of CLLibrary): The list of CL libraries this function depends on
            cl_code (str): the raw cl code for this function. This does not need to include the dependencies or
                the inclusion guard, these are added automatically here.
        """
        super(SimpleCLFunction, self).__init__()
        self._header = SimpleCLHeader(return_type, cl_function_name, parameter_list)
        self._cl_code = cl_code
        self._dependency_list = dependency_list

    @classmethod
    def construct_cl_function(cls, return_type, cl_function_name, parameter_list, cl_body, dependency_list=()):
        """A constructor that can build the full CL code from all the header parts and the CL body.

        If there are any dots in the parameter names, they will be replaced with underscores.

        Args:
            return_type (str): the CL return type of the function
            cl_function_name (string): The name of the CL function
            parameter_list (list or tuple of CLFunctionParameter): The list of parameters required for this function
            dependency_list (list or tuple of CLLibrary): The list of CL libraries this function depends on
            cl_body (str): the body of the CL code. The rest of the function call will be added by this constructor.
        """

        cl_code = '''
            {return_type} {cl_function_name}({parameters}){{
                {body}
            }}
        '''.format(return_type=return_type,
                   cl_function_name=cl_function_name,
                   parameters=', '.join('{} {}'.format(p.data_type.declaration_type, p.name.replace('.', '_'))
                                        for p in parameter_list),
                   body=cl_body)
        return cls(return_type, cl_function_name, parameter_list, cl_code, dependency_list=dependency_list)

    def get_return_type(self):
        return self._header.get_return_type()

    def get_cl_function_name(self):
        return self._header.get_cl_function_name()

    def get_parameters(self):
        return self._header.get_parameters()

    def get_cl_code(self):
        return dedent('''
            {dependencies}
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            {code}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=indent(self._get_cl_dependency_code(), ' ' * 4 * 3),
                   inclusion_guard_name='LIBRARY_FUNCTION_{}_CL'.format(self.get_cl_function_name()),
                   code=indent('\n' + self._cl_code.strip() + '\n', ' ' * 4 * 3)))

    def get_raw_cl_code(self):
        return self._cl_code

    def evaluate(self, inputs, double_precision=False):
        return CLFunctionEvaluator().evaluate(self, inputs, double_precision=double_precision)

    def _get_cl_dependency_code(self):
        """Get the CL code for all the CL code for all the dependencies.

        Returns:
            str: The CL code with the actual code.
        """
        code = ''
        for d in self._dependency_list:
            code += d.get_cl_code() + "\n"
        return code

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return type(self) != type(other)
