__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractParameterDependency(object):

    @property
    def pre_transform_code(self):
        """Some code that may be prefixed to this parameter dependency.

        Here one can put more elaborate CL code. Please make sure that additional variables are unique.

        Returns:
            str: The pre transformation code. This is prepended to the dependency function.
        """
        return ''

    @property
    def assignment_code(self):
        """Get the assignment code (including a ;).

        Returns:
            str: The assignment code.
        """
        return ''

    @property
    def fixed(self):
        """Check if this dependency fixes the parameter to the assigned value, or if it is still estimable.

        Returns:
            boolean: True if this parameter is now fixed to the dependency, or false if it is not.
        """
        return True

    @property
    def has_side_effects(self):
        """Check if the pre_transform_code from this parameter dependency has side effects to other parameters.

        Returns:
            boolean: True if this parameter has side effects (in the pre_transform_code) or does not have side-effects.
        """
        return False


class SimpleAssignment(AbstractParameterDependency):

    def __init__(self, assignment_code, fixed=True, has_side_effects=False):
        """Adds a simple parameter dependency rule for the given parameter.

        This is for one parameter, a simple one-line transformation dependency.

        Args:
            assignment_code (str): the assignment code (in CL) for this parameter
            fixed (boolean): if this parameters fixes to the assigned value or not
            has_side_effects (boolean): if this parameter changes one of the parameters before it goes into the model
                functions. Note that these side effects must be idempotent.
        """
        self._assignment = assignment_code
        self._fixed = fixed
        self._has_side_effects = has_side_effects

    @property
    def assignment_code(self):
        return self._assignment

    @property
    def fixed(self):
        return self._fixed

    @property
    def has_side_effects(self):
        return self._has_side_effects


class WeightSumToOneRule(AbstractParameterDependency):

    def __init__(self, parameter_names):
        """Adds the unity sum (parameter) dependency for the weights indicated by the given parameters.

        Parameters are given by (by <model>.<param> name). This makes sure that the given parameters sum to one before
        they are used in the model functions. Note that if you have 3 weights, you make a dependency for one of them
        with as argument the name of the other two.

        Args:
            parameter_names (list): A list with the the names of the parameters used in the dependency.
        """
        self._pre_transform_code = ''

        if len(parameter_names) < 2:
            self._assignment = '1 - ' + parameter_names[0]
            self._has_side_effects = False
        else:
            divisors = ''
            for p in parameter_names:
                divisors += p + ' /= _weight_sum;' + "\n" + "\t" * 5
            self._pre_transform_code += '''
                mot_float_type _weight_sum = ''' + ' + '.join(parameter_names) + ''';
                if(_weight_sum < 1.0){
                    _weight_sum = 1 - _weight_sum;
                }
                else{
                    ''' + divisors + '''
                    _weight_sum = 0;
                }
            '''
            self._assignment = '_weight_sum'
            self._has_side_effects = True

    @property
    def pre_transform_code(self):
        return self._pre_transform_code

    @property
    def assignment_code(self):
        return self._assignment

    @property
    def fixed(self):
        return True

    @property
    def has_side_effects(self):
        return self._has_side_effects
