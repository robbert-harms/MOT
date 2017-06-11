import logging
import pyopencl as cl
import numpy as np
from ...utils import get_float_type_def
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker


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

    def decode(self, model, data, codec):
        """Decode the given parameters using the given model.

        This transforms the data from optimization space to model space.

        Args:
            model (mot.model_interfaces.OptimizeModelInterface): The model to use
            data (ndarray): The parameters to transform to model space
            codec (mot.model_building.utils.ParameterCodec): the parameter codec to use

        Returns:
            ndarray: The array with the transformed parameters.
        """
        return self._transform_parameters(codec.get_parameter_decode_function('decodeParameters'),
                                          'decodeParameters', data, model)

    def encode(self, model, data, codec):
        """Encode the given parameters using the given model.

        This transforms the data from model space to optimization space.

        Args:
            model (mot.model_interfaces.OptimizeModelInterface): The model to use
            data (ndarray): The parameters to transform to optimization space
            codec (mot.model_building.utils.ParameterCodec): the parameter codec to use

        Returns:
            ndarray: The array with the transformed parameters.
        """
        return self._transform_parameters(codec.get_parameter_encode_function('encodeParameters'),
                                          'encodeParameters', data, model)

    def _transform_parameters(self, cl_func, cl_func_name, data, model):
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        data = np.require(data, np_dtype, requirements=['C', 'A', 'O', 'W'])
        nmr_params = data.shape[1]

        workers = self._create_workers(lambda cl_environment: _CodecWorker(
            cl_environment, self.get_compile_flags_list(model.double_precision),
            cl_func, cl_func_name, data, nmr_params, model))
        self.load_balancer.process(workers, data.shape[0])
        return data


class _CodecWorker(Worker):

    def __init__(self, cl_environment, compile_flags, cl_func, cl_func_name, data, nmr_params, model):
        super(_CodecWorker, self).__init__(cl_environment)
        self._cl_func = cl_func
        self._cl_func_name = cl_func_name
        self._data = data
        self._nmr_params = nmr_params
        self._model = model

        self._param_buf = cl.Buffer(self._cl_run_context.context,
                                    cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                    hostbuf=self._data)
        self._all_buffers = [self._param_buf]
        for data in self._model.get_data():
            self._all_buffers.append(cl.Buffer(self._cl_run_context.context,
                                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data))

        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def __del__(self):
        for buffer in self._all_buffers:
            buffer.release()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        self._kernel.transformParameterSpace(self._cl_run_context.queue, (int(nmr_problems), ), None,
                                             *self._all_buffers, global_offset=(int(range_start),))
        self._enqueue_readout(self._param_buf, self._data, range_start, range_end)

    def _get_kernel_source(self):
        kernel_param_names = ['global mot_float_type* x_global'] + \
                             self._model.get_kernel_param_names(self._cl_environment.device)

        kernel_source = ''
        kernel_source += get_float_type_def(self._model.double_precision)
        kernel_source += str(self._model.get_kernel_data_struct(self._cl_environment.device))
        kernel_source += self._cl_func
        kernel_source += '''
            __kernel void transformParameterSpace(
                ''' + ",\n".join(kernel_param_names) + '''){
                ulong gid = get_global_id(0);

                ''' + self._model.get_kernel_data_struct_initialization(self._cl_environment.device, 'data') + '''

                mot_float_type x[''' + str(self._nmr_params) + '''];

                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    x[i] = x_global[gid * ''' + str(self._nmr_params) + ''' + i];
                }

                ''' + self._cl_func_name + '''((void*)&data, x);

                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    x_global[gid * ''' + str(self._nmr_params) + ''' + i] = x[i];
                }
            }
        '''
        return kernel_source
