from mot.cl_routines.mapping.cl_function_evaluator import CLFunctionEvaluator

__author__ = 'Robbert Harms'
__date__ = '2017-08-31'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


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
            list of CLFunctionParameter: list of the parameters in this model in the same order as in the CL function"""
        raise NotImplementedError()

    def get_cl_code(self):
        """Get the function code for this function and all its dependencies.

        Returns:
            str: The CL code for inclusion in a kernel.
        """
        raise NotImplementedError()

    def evaluate(self, inputs, double_precision=False):
        """Evaluate this function for each set of given parameters.

        Given a set of input parameters, this model will be evaluated for every parameter set.

        Args:
            inputs (list of ndarray): a list with for each parameter of the model an input parameter to this function.
                Each of these input arrays must be of equal length in the first dimension.
            double_precision (boolean): if the function should be evaluated in double precision or not

        Returns:
            ndarray: a single array of the specified return type with for each parameter tuple an evaluation result
        """
        raise NotImplementedError()


class AbstractCLFunction(CLFunction):

    def __init__(self, return_type, cl_function_name, parameter_list, dependency_list=()):
        """A simple abstract implementation of a CL function.

        Most of the requirements of a CL function are satisfied by the constructor arguments, only the CL code remains.
        This needs to be overridden by an implementing subclass.

        Args:
            return_type (str): the CL return type of the function
            cl_function_name (string): The name of the CL function
            parameter_list (list or tuple of CLFunctionParameter): The list of parameters required for this function
            dependency_list (list or tuple of CLLibrary): The list of CL libraries this function depends on
        """
        self._return_type = return_type
        self._function_name = cl_function_name
        self._parameter_list = parameter_list
        self._dependency_list = dependency_list

    def get_cl_code(self):
        raise NotImplementedError()

    def get_cl_function_name(self):
        return self._function_name

    def get_return_type(self):
        return self._return_type

    def get_parameters(self):
        return self._parameter_list

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
