from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import SimpleNamedCLFunction, KernelInputArray, KernelInputAllocatedOutput
from ...cl_routines.base import CLRoutine
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CalculateDependentParameters(CLRoutine):

    def __init__(self, double_precision=False, **kwargs):
        """CL code for calculating the dependent parameters.

        Some of the models may contain parameter dependencies. We would like to return the maps for these parameters
        as well as all the other maps. Since the dependencies are specified in CL, we have to recourse to CL to
        calculate these maps.

        Args:
            double_precision (boolean): if we will use the double (True) or single floating (False) type
                for the calculations
        """
        super(CalculateDependentParameters, self).__init__(**kwargs)
        self._double_precision = double_precision

    def calculate(self, kernel_data, estimated_parameters_list, parameters_listing, dependent_parameter_names):
        """Calculate the dependent parameters

        This uses the calculated parameters in the results dictionary to run the parameters_listing in CL to obtain
        the maps for the dependent parameters.

        Args:
            kernel_data (dict[str: mot.utils.KernelInputData]): the list of additional data to load
            estimated_parameters_list (list of ndarray): The list with the one-dimensional
                ndarray of estimated parameters
            parameters_listing (str): The parameters listing in CL
            dependent_parameter_names (list of list of str): Per parameter we would like to obtain the CL name and the
                result map name. For example: (('Wball_w', 'Wball.w'),)
        Returns:
            dict: A dictionary with the calculated maps for the dependent parameters.
        """
        cl_named_func = self._get_wrapped_function(estimated_parameters_list, parameters_listing,
                                                   dependent_parameter_names)

        all_kernel_data = dict(kernel_data)
        all_kernel_data['x'] = KernelInputArray(np.dstack(estimated_parameters_list)[0, ...], ctype='mot_float_type')
        all_kernel_data['_results'] = KernelInputAllocatedOutput(
            (estimated_parameters_list[0].shape[0], len(dependent_parameter_names)), 'mot_float_type')

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(cl_named_func, all_kernel_data, estimated_parameters_list[0].shape[0],
                             double_precision=self._double_precision)

        return all_kernel_data['_results'].get_data()

    def _get_wrapped_function(self, estimated_parameters_list, parameters_listing, dependent_parameter_names):
        parameter_write_out = ''
        for i, p in enumerate([el[0] for el in dependent_parameter_names]):
            parameter_write_out += 'data->_results[' + str(i) + '] = ' + p + ";\n"

        func = '''
            void transform(mot_data_struct* data){
                mot_float_type x[''' + str(len(estimated_parameters_list)) + '''];

                for(uint i = 0; i < ''' + str(len(estimated_parameters_list)) + '''; i++){
                    x[i] = data->x[i];
                }
                ''' + parameters_listing + '''
                ''' + parameter_write_out + '''
            }
        '''
        return SimpleNamedCLFunction(func, 'transform')
