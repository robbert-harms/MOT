__author__ = 'Robbert Harms'
__date__ = "2014-06-20"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractTransformation(object):

    def __init__(self, dependencies=()):
        """The transformations define the encode and decode operations needed to build a codec.

        These objects define the basic transformation from and to model and optimization space.

        The state of the parameters is extrinsic. Calling the encode and decode functions needs a reference to a
        parameter, these should be handled by the client code.

        The transformation function may depend on other parameters. (Client code should handle the correct order
        of handling the transformations given all the dependencies.)

        Args:
            dependencies (list of (SimpleCLFunction, CLFunctionParameter) pairs): A list of (models, parameters)
                which are necessary for the operation of the transformation.
        """
        super(AbstractTransformation, self).__init__()
        self._dependencies = dependencies

    def get_cl_encode(self):
        """Get the CL encode assignment constructor

        Returns
            AssignmentConstructor: The cl code assignment constructor for encoding the parameter.
        """
        raise NotImplementedError()

    def get_cl_decode(self):
        """Get the CL decode assignment constructor

        Returns:
            AssignmentConstructor: The cl code assignment constructor for decoding the parameter.
        """
        raise NotImplementedError()

    @property
    def dependencies(self):
        """Get a list of (SimpleCLFunction, CLFunctionParameter) pairs where this transformation depends on

        Returns:
            list of tuple: A list of (SimpleCLFunction, CLFunctionParameter) tuples.
        """
        return self._dependencies


class AssignmentConstructor(object):

    def create_assignment(self, parameter_variable, lower_bound, upper_bound):
        """Create the assignment string.

        Args:
            parameter_variable (str): the name of the parameter variable holding the current value in the kernel
            lower_bound (str): the value or the name of the variable holding the value for the lower bound
            upper_bound (str): the value or the name of the variable holding the value for the upper bound

        Returns:
            str: the transformation assignment
        """
        raise NotImplementedError()


class FormatAssignmentConstructor(AssignmentConstructor):

    def __init__(self, assignment):
        """Assignment constructor that formats the given assignment template.

        This expects that the assignment string has elements like:

        * ``{parameter_variable}``: for the parameter variable
        * ``{lower_bound}``: for the lower bound
        * ``{upper_bound}``: for the upper bound
        * ``{dependency_variable_<n>}``: for the dependency variable names

        Args:
            assignment (str): the string containing the assignment template.
        """
        self._assignment = assignment

    def create_assignment(self, parameter_variable, lower_bound, upper_bound, dependency_variables=()):
        assignment = self._assignment.replace('{parameter_variable}', parameter_variable)
        assignment = assignment.replace('{lower_bound}', lower_bound)
        assignment = assignment.replace('{upper_bound}', upper_bound)

        for ind, var_name in enumerate(dependency_variables):
            assignment = assignment.replace('{dependency_variable_' + str(ind) + '}', var_name)

        return assignment


class IdentityTransform(AbstractTransformation):

    def __init__(self, *args, **kwargs):
        """The identity transform does no transformation and returns the input given."""
        super(IdentityTransform, self).__init__(*args, **kwargs)

    def get_cl_encode(self):
        return FormatAssignmentConstructor('{parameter_variable}')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('{parameter_variable}')


class ClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using the clamp function."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor('clamp((mot_float_type){parameter_variable}, '
                                           '(mot_float_type){lower_bound}, '
                                           '(mot_float_type){upper_bound})')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('clamp((mot_float_type){parameter_variable}, '
                                           '(mot_float_type){lower_bound}, '
                                           '(mot_float_type){upper_bound})')


class ScaleClampTransform(AbstractTransformation):

    def __init__(self, scale):
        """Clamps the value to the given bounds and applies a scaling to bring the parameters in sensible ranges.

        The given scaling factor should be without the scaling factor. To encode, the parameter value is multiplied
        by the scaling factor. To decode, it is divided by the scaling factor.

        Args:
            scale (float): the scaling factor by which to scale the parameter
        """
        super(ScaleClampTransform, self).__init__()
        self._scale = scale

    def get_cl_encode(self):
        return FormatAssignmentConstructor('clamp((mot_float_type){parameter_variable}, '
                                           '(mot_float_type){lower_bound}, '
                                           '(mot_float_type){upper_bound}) * ' + str(self._scale))

    def get_cl_decode(self):
        return FormatAssignmentConstructor('clamp((mot_float_type){parameter_variable} / ' + str(self._scale) + ', '
                                           '(mot_float_type){lower_bound}, '
                                           '(mot_float_type){upper_bound})')


class CosSqrClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using a cos(sqr()) transform."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor(
            'acos(clamp((mot_float_type)sqrt(fabs( ({parameter_variable} - {lower_bound}) / '
            '                                      ({upper_bound} - {lower_bound}) )), '
            '           (mot_float_type)0, (mot_float_type)1))')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('pown(cos({parameter_variable}), 2) * ' +
                                           '({upper_bound} - {lower_bound}) + {lower_bound}')


class SinSqrClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using a sin(sqr()) transform."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor(
            'asin(clamp((mot_float_type)sqrt(fabs( ({parameter_variable} - {lower_bound}) / '
            '                                       ({upper_bound} - {lower_bound}) )), '
            '           (mot_float_type)0, (mot_float_type)1))')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('pown(sin({parameter_variable}), 2) * ' +
                                           '({upper_bound} - {lower_bound}) + {lower_bound}')


class SqrClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using a sqr() transform."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor('sqrt({parameter_variable})')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('clamp((mot_float_type)({parameter_variable} * {parameter_variable}), '
                                           '      (mot_float_type){lower_bound}, '
                                           '      (mot_float_type){upper_bound})')


class SinSqrClampDependentTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between 0 and the given parameter with the sin(sqr()) transform."""

    def get_cl_encode(self):
        return FormatAssignmentConstructor('asin(sqrt(fabs(({parameter_variable} - {lower_bound}) / '
                                           '               ({dependency_variable_0} - {lower_bound}))))')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('pown(sin({parameter_variable}), 2) * {dependency_variable_0} '
                                           '        + {lower_bound}')


class AbsModXTransform(AbstractTransformation):

    def __init__(self, x, dependencies=()):
        """Create an transformation that returns the absolute modulo x value of the input."""
        super(AbsModXTransform, self).__init__(dependencies)
        self._x = x

    def get_cl_encode(self):
        return FormatAssignmentConstructor('fmod((mot_float_type)fabs({parameter_variable}), '
                                           '(mot_float_type)' + str(self._x) + ')')

    def get_cl_decode(self):
        return FormatAssignmentConstructor('fmod((mot_float_type)fabs({parameter_variable}), '
                                           '(mot_float_type)' + str(self._x) + ')')


class AbsModPiTransform(AbsModXTransform):
    def __init__(self):
        super(AbsModPiTransform, self).__init__('M_PI')
