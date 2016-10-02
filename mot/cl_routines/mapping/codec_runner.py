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

    def __init__(self, cl_environments=None, load_balancer=None, double_precision=False):
        """This class can run the codecs used to transform the parameters to and from optimization space.

        Args:
            double_precision (boolean): if we will use the double (True) or single floating (False) type for the calculations
        """
        super(CodecRunner, self).__init__(cl_environments, load_balancer)
        self._logger = logging.getLogger(__name__)
        self._double_precision = double_precision

    def decode(self, codec, data):
        """Decode the parameters.

        This transforms the data from optimization space to model space.

        Args:
            codec (AbstractCodec): The codec to use in the transformation.
            data (ndarray): The parameters to transform to model space

        Returns:
            ndarray: The array with the transformed parameters.
        """
        if len(data.shape) > 1:
            from_width = data.shape[1]
        else:
            from_width = 1

        if from_width != codec.get_nmr_parameters():
            raise ValueError("The width of the given data does not match the codec expected width.")

        return self._transform_parameters(codec.get_cl_decode_function('decodeParameters'), 'decodeParameters', data,
                                          codec.get_nmr_parameters())

    def encode(self, codec, data):
        """Encode the parameters.

        This transforms the data from model space to optimization space.

        Args:
            codec (AbstractCodec): The codec to use in the transformation.
            data (ndarray): The parameters to transform to optimization space

        Returns:
            ndarray: The array with the transformed parameters.
        """
        if len(data.shape) > 1:
            from_width = data.shape[1]
        else:
            from_width = 1

        if from_width != codec.get_nmr_parameters():
            raise ValueError("The width of the given data does not match the codec expected width.")

        return self._transform_parameters(codec.get_cl_encode_function('encodeParameters'), 'encodeParameters', data,
                                          codec.get_nmr_parameters())

    def _transform_parameters(self, cl_func, cl_func_name, data, nmr_params):
        np_dtype = np.float32
        if self._double_precision:
            np_dtype = np.float64
        data = np.require(data, np_dtype, requirements=['C', 'A', 'O', 'W'])
        rows = data.shape[0]
        workers = self._create_workers(lambda cl_environment: _CodecWorker(cl_environment, self.get_compile_flags_list(),
                                                                           cl_func, cl_func_name, data,
                                                                           nmr_params, self._double_precision))
        self.load_balancer.process(workers, rows)
        return data


class _CodecWorker(Worker):

    def __init__(self, cl_environment, compile_flags, cl_func, cl_func_name, data, nmr_params, double_precision):
        super(_CodecWorker, self).__init__(cl_environment)
        self._cl_func = cl_func
        self._cl_func_name = cl_func_name
        self._data = data
        self._nmr_params = nmr_params
        self._double_precision = double_precision

        self._param_buf = cl.Buffer(self._cl_run_context.context,
                                    cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                    hostbuf=self._data)

        self._kernel = self._build_kernel(compile_flags)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        event = self._kernel.transformParameterSpace(self._cl_run_context.queue, (int(nmr_problems), ), None,
                                                     self._param_buf, global_offset=(int(range_start),))

        return [self._enqueue_readout(self._param_buf, self._data, range_start, range_end, [event])]

    def _get_kernel_source(self):
        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._cl_func
        kernel_source += '''
            __kernel void transformParameterSpace(global mot_float_type* x_global){
                int gid = get_global_id(0);

                mot_float_type x[''' + str(self._nmr_params) + '''];

                for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    x[i] = x_global[gid * ''' + str(self._nmr_params) + ''' + i];
                }

                ''' + self._cl_func_name + '''(x);

                for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    x_global[gid * ''' + str(self._nmr_params) + ''' + i] = x[i];
                }
            }
        '''
        return kernel_source
