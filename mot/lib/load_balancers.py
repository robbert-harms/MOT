from mot.lib.utils import split_in_batches

__author__ = 'Robbert Harms'
__date__ = '2019-12-20'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class LoadBalancer:
    """Load balancers allow for fine-tuned division of labour over multiple CL devices.

    The idea is that when evaluating a kernel, the work can be split over all selected devices. The typical
    division is an even distribution, but this can be fine-tuned if one device is faster than others.
    """

    def get_division(self, cl_environments, nmr_instances):
        """Get the proposed division of labour for dividing the given number of instances over the given environments.

        Args:
            cl_environments (List[mot.lib.cl_environments.CLEnvironment]): the environments used in the load balancing.
            nmr_instances (int): the number of work instances to divide

        Returns:
            List[Tuple[int, int]]: list with (batch_start, batch_end) tuples dividing the number of instances.
        """
        raise NotImplementedError()


class EvenDistribution(LoadBalancer):
    """Evenly distribute the work over all available environments."""

    def get_division(self, cl_environments, nmr_instances):
        return list(split_in_batches(nmr_instances, nmr_batches=len(cl_environments)))


class FractionalLoad(LoadBalancer):

    def __init__(self, fractions):
        """Balance the load according to a specified fraction per devices.

        This class will round the work items such that all work gets done.

        Args:
            fractions (Tuple[float]): for each device the fraction of work that device must do.
        """
        self._fractions = fractions

    def get_division(self, cl_environments, nmr_instances):
        if len(cl_environments) != len(self._fractions):
            raise ValueError('The number of devices does not match the number of specified loads.')

        return NotImplementedError()
        # todo
