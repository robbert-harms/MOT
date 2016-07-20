import numbers

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
            dependencies (list of (CLFunction, CLFunctionParameter) pairs): A list of (models, parameters)
                which are necessary for the operation of the transformation.
        """
        super(AbstractTransformation, self).__init__()
        self._dependencies = dependencies

    def get_cl_encode(self, parameter, parameter_name, dependencies_names=()):
        """Get the CL encode string

        Args:
            parameter (CLFunctionParameter): The parameter used as context for building the encode function.
            parameter_name (str): The name of this parameter, needed to generate the cl code
            dependencies_names (list of str): For each dependency (ordered) give the name how it should be used in the
                cl code.

        Returns:
            str: The cl code that can encode the parameter.
        """
        pass

    def get_cl_decode(self, parameter, parameter_name, dependencies_names=()):
        """Get the CL decode string

        Args:
            parameter (CLFunctionParameter): The parameter used as context for building the decode function.
            parameter_name (str): The name of this parameter, needed to generate the cl code
            dependencies_names (list of str): For each dependency (ordered) give the name how it should be used in the
                cl code.

        Returns:
            str: The cl code that can decode the parameter.
        """
        pass

    @property
    def dependencies(self):
        """Get a list of (CLFunction, CLFunctionParameter) pairs where this transformation depends on

        Returns:
            A list of (CLFunction, CLFunctionParameter) pairs.
        """
        return self._dependencies


class SimpleTransformation(AbstractTransformation):
    def __init__(self, cl_encode, cl_decode):
        """Adds a simple parameter transformation rule.

        This is for one parameter, a simple one-line transformation transformation.

        Args:
            cl_to_encoded (str): the assignment that transforms the parameter from model space to encoded space.
                It should contain one {} used for the parameter_name variable.
            cl_to_model (str): the assignment that transforms the parameter from encoded space to model space.
                It should contain one {} used for the parameter_name variable.
        """
        super(SimpleTransformation, self).__init__()
        self._cl_encode = cl_encode
        self._cl_decode = cl_decode

    def get_cl_encode(self, parameter, parameter_name, dependencies_names=()):
        return self._cl_encode.format(parameter_name)

    def get_cl_decode(self, parameter, parameter_name, dependencies_names=()):
        return self._cl_decode.format(parameter_name)


class IdentityTransform(SimpleTransformation):
    def __init__(self):
        """The identity transform does no transformation and returns the input given."""
        super(IdentityTransform, self).__init__('{};', '{};')


class ClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using the clamp function."""

    def get_cl_encode(self, parameter, parameter_name, dependencies_names=()):
        return 'clamp((mot_float_type)' + parameter_name + ', (mot_float_type)' + str(parameter.lower_bound) + \
               ', (mot_float_type)' + str(parameter.upper_bound) + ');'

    def get_cl_decode(self, parameter, parameter_name, dependencies_names=()):
        return 'clamp((mot_float_type)' + parameter_name + ', (mot_float_type)' + str(parameter.lower_bound) + \
               ', (mot_float_type)' + str(parameter.upper_bound) + ');'


class CosSqrClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using a cos(sqr()) transform."""

    def get_cl_encode(self, parameter, parameter_name, dependencies_names=()):
        return 'acos(sqrt(fabs((' + parameter_name + ' - ' + \
               str(parameter.lower_bound) + ') / ' + \
               '(' + str(parameter.upper_bound) + ' - ' + str(parameter.lower_bound) + ')' + ')));'

    def get_cl_decode(self, parameter, parameter_name, dependencies_names=()):
        return 'pown(cos(' + parameter_name + '), 2) * ' + \
               '(' + str(parameter.upper_bound) + ' - ' + str(parameter.lower_bound) + ')' + \
                ' + ' + str(parameter.lower_bound) + ';'


class SinSqrClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using a sin(sqr()) transform."""

    def get_cl_encode(self, parameter, parameter_name, dependencies_names=()):
        return 'asin(sqrt(fabs((' + parameter_name + ' - ' + \
               str(parameter.lower_bound) + ') / ' + \
               '(' + str(parameter.upper_bound) + ' - ' + str(parameter.lower_bound) + ')' + ')));'

    def get_cl_decode(self, parameter, parameter_name, dependencies_names=()):
        return 'pown(sin(' + parameter_name + '), 2) * ' + \
               '(' + str(parameter.upper_bound) + ' - ' + str(parameter.lower_bound) + ')' + \
               ' + ' + str(parameter.lower_bound) + ';'


class SqrClampTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between its lower and upper bound using a sqr() transform."""

    def get_cl_encode(self, parameter, parameter_name, dependencies_names=()):
        return 'sqrt(' + parameter_name + ');'

    def get_cl_decode(self, parameter, parameter_name, dependencies_names=()):
        return 'clamp((mot_float_type)(' + parameter_name + ' * ' + parameter_name + '), (mot_float_type)' + \
               str(parameter.lower_bound) + ', (mot_float_type)' + str(parameter.upper_bound) + ');'


class SinSqrClampDependentTransform(AbstractTransformation):
    """The clamp transformation limits the parameter between 0 and the given parameter with the sin(sqr()) transform."""

    def get_cl_encode(self, parameter, parameter_name, dependencies_names=()):
        return 'asin(sqrt(fabs((' + parameter_name + ' - ' + \
               str(parameter.lower_bound) + ') / (' + \
               dependencies_names[0] + ' - ' + \
               str(parameter.lower_bound) + '))));'

    def get_cl_decode(self, parameter, parameter_name, dependencies_names=()):
        return 'pown(sin(' + parameter_name + '), 2) * ' + dependencies_names[0] + ' + ' \
                    + str(parameter.lower_bound) + ';'


class AbsModXTransform(AbstractTransformation):
    def __init__(self, x, dependencies=()):
        """Create an transformation that returns the absolute modulo x value of the input."""
        super(AbsModXTransform, self).__init__(dependencies)
        self._x = x

    def get_cl_encode(self, parameter, parameter_name, dependencies_names=()):
        return 'fmod((mot_float_type)fabs(' + parameter_name + '), (mot_float_type)' + str(self._x) + ');'

    def get_cl_decode(self, parameter, parameter_name, dependencies_names=()):
        return 'fmod((mot_float_type)fabs(' + parameter_name + '), (mot_float_type)' + str(self._x) + ');'


class CosSqrTransform(SimpleTransformation):
    def __init__(self):
        super(CosSqrTransform, self).__init__('acos(sqrt(fabs({})));', 'pown(cos({}), 2);')


class SinSqrTransform(SimpleTransformation):
    def __init__(self):
        super(SinSqrTransform, self).__init__('asin(sqrt(fabs({})));', 'pown(sin({}), 2);')


class AbsModPiTransform(AbsModXTransform):
    def __init__(self):
        super(AbsModPiTransform, self).__init__('M_PI')


class SqrTransform(SimpleTransformation):
    def __init__(self):
        super(SqrTransform, self).__init__('sqrt(fabs({}));', 'pown({}, 2);')
