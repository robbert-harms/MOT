import numbers
import warnings
from .base import AbstractFilter, AbstractFilterWorker
import numpy as np
import pyopencl as cl
from ...utils import get_cl_pragma_double, get_float_type_def

__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GaussianFilter(AbstractFilter):

    def __init__(self, size, cl_environments, load_balancer, sigma=None):
        """Create a new filterer for gaussian filtering.

        Args:
            size (int or tuple): (x, y, z, ...). Either a single dimension size for all dimensions or one value
                for each dimension of the input data to the filter function.
                Either way this value is the distance to the left and to the right of each value.
                That means that the total kernel size is the product of 1 + 2*s for each size s of each dimension.
            cl_environments: The cl environments
            load_balancer: The load balancer to use
            sigma (double or list of double): Either a single double or a list of doubles, one for each size.
                This parameter defines the sigma of the Gaussian distribution used for creating the Gaussian filtering
                kernel. If None, the sigma is calculated using size / 3.0.

        Attributes:
            size (int or tuple): (x, y, z, ...). Either a single dimension size for all dimensions or one value
                for each dimension of the input data to the filter function.
                Either way this value is the distance to the left and to the right of each value.
                That means that the total kernel size is the product of 1 + 2*s for each size s of each dimension.
            sigma (double or list of double): Either a single double or a list of doubles, one for each size.
                This parameter defines the sigma of the Gaussian distribution used for creating the Gaussian filtering
                kernel.
        """
        super(GaussianFilter, self).__init__(size, cl_environments=cl_environments, load_balancer=load_balancer)
        self.sigma = sigma

    def _get_worker(self, *args):
        """Create the worker that we will use in the computations.

        This is supposed to be overwritten by the implementing filterer.

        Returns:
            the worker object
        """
        return _GaussianFilterWorker(self, *args)


class _GaussianFilterWorker(AbstractFilterWorker):

    def calculate(self, range_start, range_end):
        volumes_to_run = [self._volumes_list[i] for i in range(len(self._volumes_list)) if range_start <= i < range_end]

        read_write_flags = self._cl_environment.get_read_write_cl_mem_flags()
        read_only_flags = self._cl_environment.get_read_only_cl_mem_flags()

        return_event = None
        for volume_name, volume in volumes_to_run:
            volume_buf = cl.Buffer(self._cl_environment.context, read_write_flags, hostbuf=volume)
            results_buf = cl.Buffer(self._cl_environment.context, read_write_flags,
                                    hostbuf=self._results_dict[volume_name])

            for dimension in range(len(self._volume_shape)):
                kernel_length = self._calculate_kernel_size_in_dimension(dimension)
                kernel_sigma = self._get_sigma_in_dimension(dimension)

                filter_kernel = self._get_1d_gaussian_kernel_array(kernel_length, kernel_sigma)
                filter_kernel_buf = cl.Buffer(self._cl_environment.context, read_only_flags, hostbuf=filter_kernel)

                kernel_source = self._get_gaussian_kernel_source(dimension)

                warnings.simplefilter("ignore")
                kernel = cl.Program(self._cl_environment.context,
                                    kernel_source).build(' '.join(self._cl_environment.compile_flags))

                if dimension % 2 == 0:
                    buffers_list = [volume_buf]
                    if self._use_mask:
                        buffers_list.append(self._mask_buf)
                    buffers_list.extend([filter_kernel_buf, results_buf])
                    results_buf_ptr = results_buf
                else:
                    buffers_list = [results_buf]
                    if self._use_mask:
                        buffers_list.append(self._mask_buf)
                    buffers_list.extend([filter_kernel_buf, volume_buf])
                    results_buf_ptr = volume_buf

                kernel.filter(self._queue, self._volume_shape, None, *buffers_list)

                if dimension == len(self._volume_shape) - 1:
                    return_event = cl.enqueue_copy(self._queue, self._results_dict[volume_name],
                                                   results_buf_ptr, is_blocking=False)

        return return_event

    def _build_kernel(self):
        pass

    def _get_gaussian_kernel_source(self, dimension):
        left_right = self._get_size_in_dimension(dimension)

        working_dim = 'dim' + str(dimension)

        kernel_source = get_cl_pragma_double()
        kernel_source += get_float_type_def(self._use_double, 'masking_float')
        kernel_source += self._get_ks_sub2ind_func(self._volume_shape)
        kernel_source += '''
            __kernel void filter(
                global masking_float* volume,
                ''' + ('global char* mask,' if self._use_mask else '') + '''
                global masking_float* filter,
                global masking_float* results
                ){

                ''' + self._get_ks_dimension_inits(len(self._volume_shape)) + '''
                const int ind = ''' + self._get_ks_sub2ind_func_call(len(self._volume_shape)) + ''';

                ''' + ('if(mask[ind] > 0){' if self._use_mask else 'if(true){') + '''
                    int filter_index = 0;
                    masking_float filtered_value = 0;

                    const int start = dim''' + str(dimension) + ''' - ''' + str(left_right) + ''';
                    const int end = dim''' + str(dimension) + ''' + ''' + str(left_right) + ''';
                    int tmp_ind = 0;

                    for(''' + working_dim + ''' = start; ''' + working_dim + ''' <= end; ''' + working_dim + '''++){
                        tmp_ind = ''' + self._get_ks_sub2ind_func_call(len(self._volume_shape)) + ''';

                        if(''' + working_dim + ''' >= 0 && ''' + working_dim + ''' < ''' + \
                            str(self._volume_shape[dimension]) + '''){
                            ''' + ('if(mask[tmp_ind] > 0){' if self._use_mask else 'if(true){') + '''
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
        if self._parent_filter.sigma is None:
            return self._get_size_in_dimension(dimension) / 3.0
        elif isinstance(self._parent_filter.sigma, numbers.Number):
            return self._parent_filter.sigma
        else:
            return self._parent_filter.sigma[dimension]

    def _get_1d_gaussian_kernel_array(self, kernel_length, sigma):
        """Generate a new gaussian kernel of length kernel_length and with the given sigma in one dimension.

        Args:
            kernel_length (integer): odd integer defining the length of the kernel (in one dimension).
            sigma (double): The sigma used in constructing the kernel.

        Returns:
            A list of the indicated length filled with a Gaussian filtering kernel.
            The kernel is normalized to sum to 1.
        """
        r = range(-int(kernel_length/2), int(kernel_length/2)+1)
        kernel = np.array([1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2.0 / (2 * sigma**2)) for x in r])

        np_dtype = np.float32
        if self._use_double:
            np_dtype = np.float64

        kernel = kernel.astype(dtype=np_dtype, order='C', copy=False)
        return kernel / sum(kernel)