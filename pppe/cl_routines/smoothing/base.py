import numbers
import warnings

import numpy as np
import pyopencl as cl

from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import WorkerConstructor
from ...tools import set_correct_cl_data_type, \
    get_write_only_cl_mem_flags, get_read_only_cl_mem_flags


__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractSmoother(AbstractCLRoutine):

    def __init__(self, size, cl_environments=None, load_balancer=None):
        """Initialize an abstract smoother routine.

        This is meant to be called by the constructor of an implementing class.

         Args:
            size (int or tuple): Either a single dimension size for all dimensions or one value
                for each dimension of the input data to the smooth function. Maximum number of dimensions is 3.
                Either way this value is the distance to the left and to the right of each value.
                That means that the total kernel size is the product of 1 + 2*s for each size s of each dimension.
            cl_environments: The cl environments
            load_balancer: The load balancer to use

        Attributes:
            size (int or tuple): (x, y, z, ...). Either a single dimension size for all dimensions or one value
                for each dimension of the input data to the smooth function.
                Either way this value is the distance to the left and to the right of each value.
                That means that the total kernel size is the product of 1 + 2*s for each size s of each dimension.
        """
        super(AbstractSmoother, self).__init__(cl_environments, load_balancer)
        self.size = size

    def smooth(self, value, mask=None):
        """Smooth the givens volumes in the given dictionary.

        If a dict is given as a value the smoothing is applied to every value in the dictionary. This can be spreaded
        over the different devices. If a single nd array is given the smoothing is performed only on one device.

        Args:
            value (dict or array like): an single array to smooth (dimensions must match the size specified in
                the constructor). Can also be a dictionary with a list of ndarrays.
            mask (array like): A single array of the same dimension as the input value. This can be used to
                mask values from being used by the smoothing routines. They are not used for smoothing other values
                and are not smoothed themselves.

        Returns:
            The same type as the input. A new set of volumes with the same keys, or a single array. All smoothed.
        """
        if isinstance(value, dict):
            if mask is None:
                mask = np.ones_like(value[value.keys()[0]], dtype=np.float64, order='C')
            return self._smooth(value, mask)
        else:
            if mask is None:
                mask = np.ones_like(value, dtype=np.float64, order='C')
            return self._smooth({'value': value}, mask)['value']

    def _smooth(self, volumes_dict, mask):
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)
        load_balancer = self.load_balancer

        results_dict = {}
        for key, value in volumes_dict.items():
            volumes_dict[key] = np.array(value, dtype=np.float64, order='C')
            results_dict[key] = np.zeros_like(volumes_dict[key], dtype=np.float64, order='C')

        volumes_list = volumes_dict.items()

        def run_transformer_cb(cl_environment, start, end, buffered_dicts):
            kernel_source = self._get_kernel_source(mask.shape, cl_environment)

            warnings.simplefilter("ignore")
            kernel = cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))
            return self._run_filter(results_dict, volumes_list, mask, start, end, cl_environment, kernel)

        worker_constructor = WorkerConstructor()
        workers = worker_constructor.generate_workers(cl_environments, run_transformer_cb)

        load_balancer.process(workers, len(volumes_list))

        return results_dict

    def _run_filter(self, results_dict, volumes_list, mask, start, end, cl_environment, kernel):
        volumes_to_run = [volumes_list[i] for i in range(len(volumes_list)) if start <= i < end]

        write_only_flags = get_write_only_cl_mem_flags(cl_environment)
        read_only_flags = get_read_only_cl_mem_flags(cl_environment)

        queue = cl_environment.get_new_queue()

        mask = mask.astype(np.int8, order='C')
        mask_buf = cl.Buffer(cl_environment.context, read_only_flags, hostbuf=mask)

        event = None
        for key, value in volumes_to_run:
            volume_buf = cl.Buffer(cl_environment.context, read_only_flags, hostbuf=value)
            results_buf = cl.Buffer(cl_environment.context, write_only_flags, hostbuf=results_dict[key])

            kernel.filter(queue, mask.shape, None, volume_buf, mask_buf, results_buf)
            event = cl.enqueue_copy(queue, results_dict[key], results_buf, is_blocking=False)

        return queue, event

    def _get_kernel_source(self, volume_shape, cl_environment):
        """Get the kernel source for this smoothing kernel.

           This should be implemented by the subclass.
        """

    def _get_size_in_dimension(self, dimension):
        if isinstance(self.size, numbers.Number):
            return self.size
        else:
            return self.size[dimension]

    def _calculate_kernel_size_in_dimension(self, dimension):
        return self._get_size_in_dimension(dimension) * 2 + 1

    def _get_ks_dimension_inits(self, nmr_dimensions):
        """Get the kernel source part for the dimension initializations"""
        s = ''
        for i in range(nmr_dimensions):
            if i > 0:
                s += "\t" * 5
            s += 'int dim' + repr(i) + ' = get_global_id(' + repr(i) + ');' + "\n"
        return s

    def _get_ks_sub2ind_func(self, volume_shape):
        """Get the kernel source part for converting array subscripts to indices"""
        s = 'int sub2ind('
        for i in range(len(volume_shape)):
            s += 'const int dim' + repr(i) + ', '
        s = s[0:-2] + '){' + "\n"
        s += "\t" * 2 + 'return '
        for i, d in enumerate(volume_shape):
            stride = ''
            for ds in volume_shape[(i + 1):]:
                stride += ' * ' + repr(ds)
            s += 'dim' + repr(i) + stride + ' + '
        s = s[0:-3] + ';' + "\n"
        s += "\t" * 1 + '}' + "\n"
        return s

    def _get_ks_sub2ind_func_call(self, nmr_dimensions):
        """Get the kernel source part for the function call for converting array subscripts to indices"""
        s = 'sub2ind('
        for i in range(nmr_dimensions):
            s += 'dim' + repr(i) + ', '
        return s[0:-2] + ')'

    def _calculate_length(self, nmr_dimensions):
        """Calculate the length of the array given the number of dimensions.

        The kernel size is determined by the global size tuple. For each dimension this specifies the number of
        values we look to the right and to the left to calculate the new value. This means that the total kernel size
        of all dimensions together is the multiplication of 2 * n + 1 for each dimension: left + right + current.

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
            if i > 0:
                s += "\t" * 6
            s += 'int dim' + repr(i) + '_start = max(0, dim' + repr(i) + ' - ' + repr(self._get_size_in_dimension(i)) \
                 + ');' + "\n"
            s += "\t" * 6
            s += 'int dim' + repr(i) + '_end = min(' + repr(d) + ', dim' + repr(i) + ' + ' + \
                 repr(self._get_size_in_dimension(i)) + ' + 1);' + "\n"
        return s