import logging
import numbers
import numpy as np
import pyopencl as cl
from mot.cl_routines.base import CLRoutine
from mot.load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractFilter(CLRoutine):

    def __init__(self, size, cl_environments=None, load_balancer=None):
        """Initialize the filter routine.

        Args:
            size (int or tuple): Either a single dimension size for all dimensions or one value
                for each dimension of the input data to the filter function. Maximum number of dimensions is 3.
                Either way this value is the distance to the left and to the right of each value.
                That means that the total kernel size is the product of 1 + 2*s for each size s of each dimension.
            cl_environments: The cl environments
            load_balancer: The load balancer to use

        Attributes:
            size (int or tuple): (x, y, z, ...). Either a single dimension size for all dimensions or one value
                for each dimension of the input data to the filter function.
                Either way this value is the distance to the left and to the right of each value.
                That means that the total kernel size is the product of 1 + 2*s for each size s of each dimension.
        """
        super(AbstractFilter, self).__init__(cl_environments=cl_environments, load_balancer=load_balancer)
        self.size = size
        self._logger = logging.getLogger(__name__)

    def filter(self, value, mask=None, double_precision=False, nmr_of_times=1):
        """Filter the given volumes in the given dictionary.

        If a dict is given as a value the filtering is applied to every value in the dictionary. This can be spread
        over the different devices. If a single ndarray is given the filtering is performed only on one device.

        Args:
            value (dict or array like): an single array to filter (dimensions must match the size specified in
                the constructor). Can also be a dictionary with a multitude of ndarrays.
            mask (array like): A single array of the same dimension as the input value. This can be used to
                mask values from being used by the filtering routines. They are not used for filtering other values
                and are not filtered themselves.
            double_precision (boolean): if we will use double or float
            nmr_of_times (int): how many times we would like to repeat the filtering (per input volume).

        Returns:
            The same type as the input. A new set of volumes with the same keys, or a single array. All filtered.
        """
        if nmr_of_times < 1:
            nmr_of_times = 1

        if nmr_of_times == 1:
            self._logger.info('Applying filtering with filter {0}'.format(self.__class__.__name__))
            if isinstance(value, dict):
                return self._filter(value, mask, double_precision)
            else:
                return self._filter({'value': value}, mask, double_precision)['value']
        else:
            filtered = self.filter(value, mask, double_precision, 1)
            return self.filter(filtered, mask, double_precision, nmr_of_times - 1)

    def _filter(self, volumes_dict, mask, double_precision):
        results_dict = {}

        np_dtype = np.float32
        if double_precision:
            np_dtype = np.float64

        for key, value in volumes_dict.items():
            if len(value.shape) > 3 and value.shape[3] > 1:
                raise ValueError('The given volume {} is a 4d volume with a 4th dimension >1. We can not use this.')

            volumes_dict[key] = value.astype(dtype=np_dtype, copy=False, order='C')
            results_dict[key] = np.zeros_like(volumes_dict[key], dtype=np_dtype, order='C')

        if mask is not None:
            mask = mask.astype(np.int8, order='C', copy=True)

        volumes_list = list(volumes_dict.items())
        workers = self._create_workers(self._get_worker_generator(self, results_dict, volumes_list, mask,
                                                                  double_precision))
        self._load_balancer.process(workers, len(volumes_list))

        return results_dict

    def _get_worker_generator(self, *args):
        """Generate the worker generator callback for the function _create_workers()

        This is supposed to be overwritten by the implementing filterer.

        Returns:
            the python callback for generating the worker
        """
        return lambda cl_environment: AbstractFilterWorker(cl_environment, self.get_compile_flags_list(args[-1]), *args)


class AbstractFilterWorker(Worker):

    def __init__(self, cl_environment, compile_flags, parent_filter, results_dict,
                 volumes_list, mask, double_precision):
        """Create a filter worker.

        Args:
            nmr_of_times: the number of times we want to apply the filter per dataset.
        """

        super(AbstractFilterWorker, self).__init__(cl_environment)
        self._parent_filter = parent_filter
        self._size = self._parent_filter.size
        self._results_dict = results_dict
        self._volumes_list = volumes_list
        self._volume_shape = volumes_list[0][1].shape[0:3]
        self._double_precision = double_precision
        self._mask = mask
        self._logger = logging.getLogger(__name__)

        if mask is None:
            self._use_mask = False
        else:
            self._use_mask = True
            self._mask_buf = cl.Buffer(self._cl_run_context.context,
                                       cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                       hostbuf=self._mask)

        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def calculate(self, range_start, range_end):
        volumes_to_run = [self._volumes_list[i] for i in range(len(self._volumes_list)) if range_start <= i < range_end]

        volume_buf = cl.Buffer(self._cl_run_context.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=volumes_to_run[0][1])

        results_buf = cl.Buffer(self._cl_run_context.context,
                                cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                hostbuf=self._results_dict[volumes_to_run[0][0]])

        for key, value in volumes_to_run:
            cl.enqueue_copy(self._cl_run_context.queue, volume_buf, value, is_blocking=False)
            cl.enqueue_copy(self._cl_run_context.queue, results_buf, self._results_dict[key], is_blocking=False)

            buffers = [volume_buf]
            if self._use_mask:
                buffers.append(self._mask_buf)
            buffers.append(results_buf)

            self._kernel.filter(self._cl_run_context.queue, self._volume_shape, None, *buffers)
            cl.enqueue_copy(self._cl_run_context.queue, self._results_dict[key], results_buf, is_blocking=False)

    def _get_kernel_source(self):
        """Get the kernel source for this filtering kernel.

        This should be implemented by the subclass.
        """
        raise NotImplementedError()

    def _get_size_in_dimension(self, dimension):
        if isinstance(self._size, numbers.Number):
            return self._size
        else:
            return self._size[dimension]

    def _calculate_kernel_size_in_dimension(self, dimension):
        return self._get_size_in_dimension(dimension) * 2 + 1

    def _get_ks_dimension_inits(self, nmr_dimensions):
        """Get the kernel source part for the dimension initializations"""
        s = ''
        for i in range(nmr_dimensions):
            s += 'long dim' + str(i) + ' = get_global_id(' + str(i) + ');' + "\n"
        return s

    def _get_ks_sub2ind_func(self, volume_shape):
        """Get the kernel source part for converting array subscripts to indices"""
        s = 'long sub2ind('
        for i in range(len(volume_shape)):
            s += 'const long dim' + str(i) + ', '
        s = s[0:-2] + '){' + "\n"
        s += 'return '
        for i, d in enumerate(volume_shape):
            stride = ''
            for ds in volume_shape[(i + 1):]:
                stride += ' * ' + str(ds)
            s += 'dim' + str(i) + stride + ' + '
        s = s[0:-3] + ';' + "\n"
        s += '}' + "\n"
        return s

    def _get_ks_sub2ind_func_call(self, nmr_dimensions):
        """Get the kernel source part for the function call for converting array subscripts to indices"""
        s = 'sub2ind('
        for i in range(nmr_dimensions):
            s += 'dim' + str(i) + ', '
        return s[0:-2] + ')'

    def _calculate_length(self, nmr_dimensions):
        """Calculate the length of the array given the number of dimensions.

        The kernel size is determined by the global size tuple. For each dimension this specifies the number of
        values we look to the right and to the left to calculate the new value. This means that the total kernel size
        of all dimensions together is the multiplication of 2 * n + 1 for each dimension: left + right + current.

        Args:
            nmr_dimensions (int): the number of dimensions
        """
        n = 1
        for dimension in range(nmr_dimensions):
            s = self._get_size_in_dimension(dimension)
            n *= (s * 2 + 1)
        return n

    def _get_ks_dimension_sizes(self, volume_shape):
        """Get the kernel source for the start and end of each of the dimensions"""
        s = ''
        for i, d in enumerate(volume_shape):
            s += 'long dim' + str(i) + '_start = max((long)0, dim' + str(i) + ' - ' + \
                 str(self._get_size_in_dimension(i)) + ');' + "\n"
            s += 'long dim' + str(i) + '_end = min((long)' + str(d) + ', dim' + str(i) + ' + ' + \
                 str(self._get_size_in_dimension(i)) + ' + 1);' + "\n"
        return s
