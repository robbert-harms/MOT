from mot.lib.cl_function import CLFunction, SimpleCLFunction
from mot.lib.utils import split_cl_function

__author__ = 'Robbert Harms'
__date__ = '2018-08-01'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


return_code_labels = {
    0: ['default', 'no return code specified'],
    1: ['found zero', 'sum of squares/evaluation below underflow limit'],
    2: ['converged', 'the relative error in the sum of squares/evaluation is at most tol'],
    3: ['converged', 'the relative error of the parameter vector is at most tol'],
    4: ['converged', 'both errors are at most tol'],
    5: ['trapped', 'by degeneracy; increasing epsilon might help'],
    6: ['exhausted', 'number of function calls exceeding preset patience'],
    7: ['failed', 'ftol<tol: cannot reduce sum of squares any further'],
    8: ['failed', 'xtol<tol: cannot improve approximate solution any further'],
    9: ['failed', 'gtol<tol: cannot improve approximate solution any further'],
    10: ['NaN', 'Function value is not-a-number or infinite'],
    11: ['exhausted', 'temperature decreased to 0.0']
}


class OptimizeResults(dict):
    """Represents the optimization results.

    Attributes:
        x (ndarray): the optimized parameter maps, an (d, p) array with for d problems a value for every p parameters
        status (ndarray): the return codes, an (d,) vector with for d problems the status return code
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class ConstraintFunction(CLFunction):
    """These functions are meant to be provided to the optimization routines.

    If provided to the optimization routines, they should hold a CL function with the signature:

    .. code-block:: c

        void <func_name>(local const mot_float_type* const x,
                         void* data,
                         local mot_float_type* constraints);

    Although this is not enforced for general usage of this class.

    Since the number of constraints in the ``constraints`` array is variable, this class additionally specifies the
    number of constraints using the method :meth:`get_nmr_constraints`.
    """

    def get_nmr_constraints(self):
        """Get the number of constraints defined in this function.

        Returns:
            int: the number of constraints defined in this function.
        """
        raise NotImplementedError()


class SimpleConstraintFunction(SimpleCLFunction, ConstraintFunction):

    def __init__(self, *args, nmr_constraints=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._nmr_constraints = nmr_constraints
        if nmr_constraints is None:
            raise ValueError('Number of constraints not defined.')

    @classmethod
    def from_string(cls, cl_function, dependencies=(), nmr_constraints=None):
        """Parse the given CL function into a SimpleCLFunction object.

        Args:
            cl_function (str): the function we wish to turn into an object
            dependencies (list or tuple of CLLibrary): The list of CL libraries this function depends on

        Returns:
            SimpleCLFunction: the CL data type for this parameter declaration
        """
        return_type, function_name, parameter_list, body = split_cl_function(cl_function)
        return SimpleConstraintFunction(return_type, function_name, parameter_list, body, dependencies=dependencies,
                                        nmr_constraints=nmr_constraints)

    def get_nmr_constraints(self):
        return self._nmr_constraints
