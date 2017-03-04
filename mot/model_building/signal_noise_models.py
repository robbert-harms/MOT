from mot.model_building.cl_functions.parameters import FreeParameter
from mot.model_building.cl_functions.base import ModelFunction
from mot.model_building.parameter_functions.transformations import CosSqrClampTransform
from mot.cl_data_type import CLDataType

__author__ = 'Robbert Harms'
__date__ = "2014-08-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SignalNoiseModel(ModelFunction):

    def __init__(self, name, cl_function_name, parameter_list, dependency_list=()):
        """Signal noise models can add noise to the signal resulting from the model.

        They require the signal resulting from the model and zero or more parameters and they return a new signal
        with noise added.
        """
        super(SignalNoiseModel, self).__init__(name, cl_function_name, parameter_list, dependency_list=dependency_list)

    def get_signal_function(self, fname='signalNoiseModel'):
        """Get the signal function that adds the noise to the signal function.

        Args:
            fname (str, optional): The function name of the function in OpenCL.

        Returns:
            str: A function with signature:

            .. code-block:: c

                mot_float_type fname(const mot_float_type signal, <noise model parameters ...>);

            For example, if the noise model has only one parameter 'sigma' the function should look like:

            .. code-block:: c

                mot_float_type fname(const mot_float_type signal, const mot_float_type sigma);

            The CL function should return a single mot_float_type that represents the signal with the signal noise
                added to it.
        """


class JohnsonSignalNoise(SignalNoiseModel):

    def __init__(self):
        """Johnson noise adds noise to the signal using the formula:

        .. code-block:: c

            sqrt(signal^2 + eta^2)

        """
        super(JohnsonSignalNoise, self).__init__(
            'JohnsonNoise',
            'johnsonNoiseModel',
            (FreeParameter(CLDataType.from_string('mot_float_type'), 'eta', False, 0.1, 0, 100,
                           parameter_transform=CosSqrClampTransform()),), ())

    def get_signal_function(self, fname='signalNoiseModel'):
        return '''
            mot_float_type ''' + fname + '''(const mot_float_type signal, const mot_float_type eta){
                return sqrt((signal * signal) + (eta * eta));
            }
        '''
