import numbers
import numpy as np
import pyopencl as cl
from mot.cl_routines.filters.base import AbstractFilter, AbstractFilterWorker
from mot.utils import get_float_type_def


__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GaussianFilter(AbstractFilter):

    def __init__(self, size, sigma=None, cl_environments=None, load_balancer=None):
        """Initialize a Gaussian filter.

        Args:
            size (int or tuple): (x, y, z, ...). Either a single dimension size for all dimensions or one value
                for each dimension of the input data to the filter function.
                Either way this value is the distance to the left and to the right of each value.
                That means that the total kernel size is the product of 1 + 2*s for each size s of each dimension.
            sigma (double or list of double): Either a single double or a list of doubles, one for each size.
                This parameter defines the sigma of the Gaussian distribution used for creating the Gaussian filtering
                kernel. If None, the sigma is calculated using size / 3.0.
        """
        super(GaussianFilter, self).__init__(size, cl_environments=cl_environments, load_balancer=load_balancer)
        self.sigma = sigma

    def _get_worker_generator(self, *args):
        return lambda cl_environment: _GaussianFilterWorker(
            cl_environment, self.get_compile_flags_list(double_precision=True), *args)


class _GaussianFilterWorker(AbstractFilterWorker):

    def calculate(self, range_start, range_end):
        volumes_to_run = [self._volumes_list[i] for i in range(len(self._volumes_list)) if range_start <= i < range_end]

        volume_buf = cl.Buffer(self._cl_run_context.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=volumes_to_run[0][1])

        results_buf = cl.Buffer(self._cl_run_context.context,
                                cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                hostbuf=self._results_dict[volumes_to_run[0][0]])

        for volume_name, volume in volumes_to_run:
            cl.enqueue_copy(self._cl_run_context.queue, volume_buf, volume, is_blocking=False)
            cl.enqueue_copy(self._cl_run_context.queue, results_buf, self._results_dict[volume_name], is_blocking=False)

            for dimension in range(len(self._volume_shape)):
                kernel_length = self._calculate_kernel_size_in_dimension(dimension)
                kernel_sigma = self._get_sigma_in_dimension(dimension)

                filter_kernel = self._get_1d_gaussian_kernel_array(kernel_length, kernel_sigma)
                filter_kernel_buf = cl.Buffer(self._cl_run_context.context,
                                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                              hostbuf=filter_kernel)

                kernel_source = self._get_gaussian_kernel_source(dimension)
                kernel = cl.Program(self._cl_run_context.context, kernel_source).build()

                if dimension % 2 == 0:
                    buffers_list = self._list_all_buffers(volume_buf, filter_kernel_buf, results_buf)
                    results_buf_ptr = results_buf
                else:
                    buffers_list = self._list_all_buffers(results_buf, filter_kernel_buf, volume_buf)
                    results_buf_ptr = volume_buf

                kernel.filter(self._cl_run_context.queue, self._volume_shape, None, *buffers_list)

                if dimension == len(self._volume_shape) - 1:
                    cl.enqueue_copy(self._cl_run_context.queue, self._results_dict[volume_name],
                                    results_buf_ptr, is_blocking=False)

    def _build_kernel(self, compile_flags=()):
        pass

    def _list_all_buffers(self, input_buffer, filter_kernel_buffer, output_buffer):
        """Helper function of calculate().

        This creates a list with buffers and inserts the mask buffer if needed.
        """
        buffers_list = [input_buffer]
        if self._use_mask:
            buffers_list.append(self._mask_buf)
        buffers_list.extend([filter_kernel_buffer, output_buffer])
        return buffers_list

    def _get_gaussian_kernel_source(self, dimension):
        left_right = self._get_size_in_dimension(dimension)

        working_dim = 'dim' + str(dimension)

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._get_ks_sub2ind_func(self._volume_shape)
        kernel_source += '''
            __kernel void filter(
                global mot_float_type* volume,
                ''' + ('global char* mask,' if self._use_mask else '') + '''
                global mot_float_type* filter,
                global mot_float_type* results
                ){

                ''' + self._get_ks_dimension_inits(len(self._volume_shape)) + '''
                const long ind = ''' + self._get_ks_sub2ind_func_call(len(self._volume_shape)) + ''';

                ''' + ('if(mask[ind] > 0){' if self._use_mask else 'if(true){') + '''
                    long filter_index = 0;
                    mot_float_type filtered_value = 0;

                    const long start = dim''' + str(dimension) + ''' - ''' + str(left_right) + ''';
                    const long end = dim''' + str(dimension) + ''' + ''' + str(left_right) + ''';
                    long tmp_ind = 0;

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
        if self._double_precision:
            np_dtype = np.float64

        kernel = kernel.astype(dtype=np_dtype, order='C', copy=False)
        return kernel / sum(kernel)
