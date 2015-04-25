import warnings
import pyopencl as cl
from ...utils import get_cl_double_extension_definer, \
    get_read_write_cl_mem_flags, set_correct_cl_data_type
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import WorkerConstructor


__author__ = 'Robbert Harms'
__date__ = "2014-05-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CodecRunner(AbstractCLRoutine):

    def __init__(self, cl_environments=None, load_balancer=None):
        """This class can run the codecs used to transform the parameters to and from optimization space."""
        super(CodecRunner, self).__init__(cl_environments, load_balancer)

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
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)

        rows = data.shape[0]
        data = set_correct_cl_data_type(data)

        def run_transformer_cb(cl_environment, start, end, buffered_dicts):
            warnings.simplefilter("ignore")
            kernel_source = self._get_kernel_source(cl_func, cl_func_name, nmr_params, cl_environment)
            kernel = cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))
            return self._run_transformer(data, start, end, cl_environment, kernel)

        worker_constructor = WorkerConstructor()
        workers = worker_constructor.generate_workers(cl_environments, run_transformer_cb)

        self.load_balancer.process(workers, rows)
        return data

    def _run_transformer(self, parameters, start, end, cl_environment, kernel):
        queue = cl_environment.get_new_queue()
        nmr_problems = end - start
        read_write_flags = get_read_write_cl_mem_flags(cl_environment)

        param_buf = cl.Buffer(cl_environment.context, read_write_flags, hostbuf=parameters[start:end, :])

        kernel.transformParameterSpace(queue, (int(nmr_problems), ), None, param_buf)
        event = cl.enqueue_copy(queue, parameters[start:end, :], param_buf, is_blocking=False)

        return queue, event

    def _get_kernel_source(self, cl_func, cl_func_name, nmr_params, environment):
        kernel_source = get_cl_double_extension_definer(environment.platform)
        kernel_source += cl_func
        kernel_source += '''
            __kernel void transformParameterSpace(global double* x_global){
                int gid = get_global_id(0);

                double x[''' + repr(nmr_params) + '''];

                for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                    x[i] = x_global[gid * ''' + repr(nmr_params) + ''' + i];
                }

                ''' + cl_func_name + '''(x);

                for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                    x_global[gid * ''' + repr(nmr_params) + ''' + i] = x[i];
                }
            }
        '''
        return kernel_source