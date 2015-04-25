import numbers
import warnings
from .base import AbstractSmoother
import numpy as np
import pyopencl as cl
from ...load_balance_strategies import WorkerConstructor
from ...utils import get_write_only_cl_mem_flags, get_read_only_cl_mem_flags, get_cl_double_extension_definer, \
    get_read_write_cl_mem_flags


__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GaussianSmoother(AbstractSmoother):

    def __init__(self, size, cl_environments=None, load_balancer=None, sigma=None):
        """Create a new smoother for gaussian smoothing.

        Args:
            size (int or tuple): (x, y, z, ...). Either a single dimension size for all dimensions or one value
                for each dimension of the input data to the smooth function.
                Either way this value is the distance to the left and to the right of each value.
                That means that the total kernel size is the product of 1 + 2*s for each size s of each dimension.
            cl_environments: The cl environments
            load_balancer: The load balancer to use
            sigma (double or list of double): Either a single double or a list of doubles, one for each size.
                This parameter defines the sigma of the Gaussian distribution used for creating the Gaussian smoothing
                kernel. If None, the sigma is calculated using size / 3.0.

        Attributes:
            size (int or tuple): (x, y, z, ...). Either a single dimension size for all dimensions or one value
                for each dimension of the input data to the smooth function.
                Either way this value is the distance to the left and to the right of each value.
                That means that the total kernel size is the product of 1 + 2*s for each size s of each dimension.
            sigma (double or list of double): Either a single double or a list of doubles, one for each size.
                This parameter defines the sigma of the Gaussian distribution used for creating the Gaussian smoothing
                kernel.
        """
        super(GaussianSmoother, self).__init__(size, cl_environments=cl_environments, load_balancer=load_balancer)
        self.sigma = sigma

    def _smooth(self, volumes_dict, mask):
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)
        load_balancer = self.load_balancer

        results_dict = {}
        for key, value in volumes_dict.items():
            volumes_dict[key] = np.array(value, dtype=np.float64, order='C')
            results_dict[key] = np.zeros_like(volumes_dict[key], dtype=np.float64, order='C')

        volumes_list = volumes_dict.items()

        def run_transformer_cb(cl_environment, start, end, buffered_dicts):
            return self._run_all_filter_steps(results_dict, volumes_list, mask, start, end, cl_environment)

        worker_constructor = WorkerConstructor()
        workers = worker_constructor.generate_workers(cl_environments, run_transformer_cb)

        load_balancer.process(workers, len(volumes_list))

        return results_dict

    def _run_all_filter_steps(self, results_dict, volumes_list, mask, start, end, cl_environment):
        volumes_to_run = [volumes_list[i] for i in range(len(volumes_list)) if start <= i < end]

        read_write_flags = get_read_write_cl_mem_flags(cl_environment)
        read_only_flags = get_read_only_cl_mem_flags(cl_environment)

        queue = cl_environment.get_new_queue()

        mask = mask.astype(np.int8, order='C')
        mask_buf = cl.Buffer(cl_environment.context, read_only_flags, hostbuf=mask)

        return_event = None
        for volume_name, volume in volumes_to_run:
            volume_buf = cl.Buffer(cl_environment.context, read_write_flags, hostbuf=volume)
            results_buf = cl.Buffer(cl_environment.context, read_write_flags, hostbuf=results_dict[volume_name])

            if volume.shape != mask.shape:
                raise ValueError('The shape of the mask and the volumes should match exactly.')

            for dimension in range(len(mask.shape)):
                kernel_length = self._calculate_kernel_size_in_dimension(dimension)
                kernel_sigma = self._get_sigma_in_dimension(dimension)

                smooth_kernel = self._get_1d_gaussian_kernel(kernel_length, kernel_sigma)
                smooth_kernel_buf = cl.Buffer(cl_environment.context, read_only_flags, hostbuf=smooth_kernel)

                kernel_source = self._get_gaussian_kernel_source(mask.shape, cl_environment, dimension)

                warnings.simplefilter("ignore")
                kernel = cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))

                if dimension % 2 == 0:
                    buffers_list = [volume_buf, mask_buf, smooth_kernel_buf, results_buf]
                    results_buf_ptr = results_buf
                else:
                    buffers_list = [results_buf, mask_buf, smooth_kernel_buf, volume_buf]
                    results_buf_ptr = volume_buf

                kernel.filter(queue, mask.shape, None, *buffers_list)

                if dimension == len(mask.shape) - 1:
                    return_event = cl.enqueue_copy(queue, results_dict[volume_name], results_buf_ptr, is_blocking=False)

        return queue, return_event

    def _get_gaussian_kernel_source(self, volume_shape, cl_environment, dimension):
        left_right = self._get_size_in_dimension(dimension)

        working_dim = 'dim' + repr(dimension)

        kernel_source = get_cl_double_extension_definer(cl_environment.platform)
        kernel_source += self._get_ks_sub2ind_func(volume_shape)
        kernel_source += '''
            __kernel void filter(
                global double* volume,
                global char* mask,
                global double* filter,
                global double* results
                ){

                ''' + self._get_ks_dimension_inits(len(volume_shape)) + '''
                const int ind = ''' + self._get_ks_sub2ind_func_call(len(volume_shape)) + ''';

                if(mask[ind] > 0){
                    int filter_index = 0;
                    double filtered_value = 0;

                    const int start = dim''' + repr(dimension) + ''' - ''' + repr(left_right) + ''';
                    const int end = dim''' + repr(dimension) + ''' + ''' + repr(left_right) + ''';
                    int tmp_ind = 0;

                    for(''' + working_dim + ''' = start; ''' + working_dim + ''' <= end; ''' + working_dim + '''++){
                        tmp_ind = ''' + self._get_ks_sub2ind_func_call(len(volume_shape)) + ''';

                        if(''' + working_dim + ''' >= 0 && ''' + working_dim + ''' < ''' + \
                            repr(volume_shape[dimension]) + '''){
                            if(mask[tmp_ind] > 0){
                                filtered_value += filter[filter_index] * volume[tmp_ind];
                            }
                        }
                        filter_index++;
                    }

                    results[ind] = filtered_value;
                }
            }
        '''
        return kernel_source

    def _get_sigma_in_dimension(self, dimension):
        if self.sigma is None:
            return self._get_size_in_dimension(dimension) / 3.0
        elif isinstance(self.sigma, numbers.Number):
            return self.sigma
        else:
            return self.sigma[dimension]

    def _get_1d_gaussian_kernel(self, kernel_length, sigma):
        """Generate a new gaussian kernel of length kernel_length and with the given sigma in one dimension.

        Args:
            kernel_length (integer): odd integer defining the length of the kernel (in one dimension).
            sigma (double): The sigma used in constructing the kernel.

        Returns:
            A list of the indicated length filled with a Gaussian smoothing kernel.
            The kernel is normalized to sum to 1.
        """
        r = range(-int(kernel_length/2), int(kernel_length/2)+1)
        kernel = np.array([1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2.0 / (2 * sigma**2)) for x in r])
        kernel = kernel.astype(dtype=np.float64, order='C')
        return kernel / sum(kernel)