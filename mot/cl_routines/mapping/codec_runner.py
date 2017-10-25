import logging
from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import SimpleNamedCLFunction, KernelInputArray
from ...cl_routines.base import CLRoutine


__author__ = 'Robbert Harms'
__date__ = "2014-05-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CodecRunner(CLRoutine):

    def __init__(self, **kwargs):
        """This class can run the parameter encoding and decoding transformations.

        These transformations are used to transform the parameters to and from optimization space.
        """
        super(CodecRunner, self).__init__(**kwargs)
        self._logger = logging.getLogger(__name__)

    def decode(self, parameters, kernel_data, codec, double_precision=False):
        """Decode the given parameters using the given model.

        This transforms the data from optimization space to model space.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.utils.KernelInputData]): the additional data to load
            codec (mot.model_building.utils.ParameterCodec): the parameter codec to use
            double_precision (boolean): if we are running in double precision or not

        Returns:
            ndarray: The array with the transformed parameters.
        """
        return self._transform_parameters(codec.get_parameter_decode_function('decodeParameters'),
                                          'decodeParameters', parameters, kernel_data, double_precision)

    def encode(self, parameters, kernel_data, codec, double_precision=False):
        """Encode the given parameters using the given model.

        This transforms the data from model space to optimization space.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.utils.KernelInputData]): the additional data to load
            codec (mot.model_building.utils.ParameterCodec): the parameter codec to use
            double_precision (boolean): if we are running in double precision or not

        Returns:
            ndarray: The array with the transformed parameters.
        """
        return self._transform_parameters(codec.get_parameter_encode_function('encodeParameters'),
                                          'encodeParameters', parameters, kernel_data, double_precision)

    def encode_decode(self, parameters, kernel_data, codec, double_precision=False):
        """First apply an encoding operation and then apply a decoding operation again.

        This can be used to enforce boundary conditions in the parameters.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.utils.KernelInputData]): the additional data to load
            codec (mot.model_building.utils.ParameterCodec): the parameter codec to use
            double_precision (boolean): if we are running in double precision or not

        Returns:
            ndarray: The array with the transformed parameters.
        """
        func_name = 'encode_decode'
        func = ''
        func += codec.get_parameter_encode_function('encodeParameters')
        func += codec.get_parameter_decode_function('decodeParameters')
        func += '''
            void ''' + func_name + '''(mot_data_struct* data, mot_float_type* x){
                encodeParameters(data, x);
                decodeParameters(data, x);
            }
        '''
        return self._transform_parameters(func, func_name, parameters, kernel_data, double_precision)

    def _transform_parameters(self, cl_func, cl_func_name, parameters, kernel_data, double_precision):
        cl_named_func = self._get_codec_function_wrapper(cl_func, cl_func_name, parameters.shape[1])

        all_kernel_data = dict(kernel_data)
        all_kernel_data['x'] = KernelInputArray(parameters, ctype='mot_float_type', is_writable=True)

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(cl_named_func, all_kernel_data, parameters.shape[0], double_precision=double_precision)
        return all_kernel_data['x'].get_data()

    def _get_codec_function_wrapper(self, cl_func, cl_func_name, nmr_params):
        func_name = 'transformParameterSpace'
        func = cl_func
        func += '''
             void ''' + func_name + '''(mot_data_struct* data){
                mot_float_type x[''' + str(nmr_params) + '''];
                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    x[i] = data->x[i];
                }

                ''' + cl_func_name + '''(data, x);

                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    data->x[i] = x[i];
                }
            }
        '''
        return SimpleNamedCLFunction(func, func_name)
