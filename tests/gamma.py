import unittest
from scipy.stats import gamma
import numpy as np
import pyopencl as cl
from numpy.testing import assert_allclose

from mot.library_functions import GammaFunctions
from mot.utils import cartesian


class test_GammaFunctions(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(test_GammaFunctions, self).__init__(*args, **kwargs)

    def test_gamma_cdf(self):
        test_params = self._gamma_cdf_params().astype(dtype=np.float32)

        python_results = self._calculate_python(test_params)
        cl_results = self._calculate_cl(test_params)

        assert_allclose(np.nan_to_num(python_results), np.nan_to_num(cl_results), atol=1e-5, rtol=1e-5)

    def _calculate_cl(self, test_params):
        test_params = test_params.astype(np.float64)
        src = self._get_kernel_source(test_params.shape[0])
        results = np.zeros(test_params.shape[0])
        self._run_kernel(src, test_params, results)
        return results

    def _run_kernel(self, src, input_args, results):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, src).build()

        mf = cl.mem_flags
        input_args_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_args)
        output_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=results)
        buffers = [input_args_buffer, output_buffer]

        kernel_event = prg.test_gamma_cdf(queue, (input_args.shape[0],), None, *buffers)
        self._enqueue_readout(queue, output_buffer, results, 0, input_args.shape[0], [kernel_event])

    def _enqueue_readout(self, queue, buffer, host_array, range_start, range_end, wait_for):
        nmr_problems = range_end - range_start
        return cl.enqueue_map_buffer(
            queue, buffer, cl.map_flags.READ, range_start * host_array.strides[0],
            (nmr_problems, ) + host_array.shape[1:], host_array.dtype, order="C", wait_for=wait_for,
            is_blocking=False)[1]

    def _get_kernel_source(self, nmr_problems):
        src = ''
        src += GammaFunctions().get_cl_code()
        src += '''
            __kernel void test_gamma_cdf(global double input_args[''' + str(nmr_problems) + '''][3],
                                         global double output[''' + str(nmr_problems) + ''']){

                uint gid = get_global_id(0);

                double shape = input_args[gid][0];
                double scale = input_args[gid][1];
                double x = input_args[gid][2];
                
                output[gid] = gamma_cdf(shape, scale, x);
            }
        '''
        return src

    def _gamma_cdf_params(self):
        """Get some test params to test the Gamma CDF
        """
        shapes = np.arange(0.1, 5, 0.2)
        scales = np.arange(0.1, 5, 0.2)
        xs = np.arange(0.1, 10, 0.2)
        arr = cartesian([shapes, scales, xs])
        return arr

    def _calculate_python(self, input_params):
        results = np.zeros(input_params.shape[0])

        for ind in range(input_params.shape[0]):
            shape = input_params[ind, 0]
            scale = input_params[ind, 1]
            x = input_params[ind, 2]

            results[ind] = gamma.cdf(x, shape, scale=scale)
        return results
