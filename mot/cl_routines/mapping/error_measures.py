import numpy as np
from ...cl_routines.base import AbstractCLRoutine


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ErrorMeasures(AbstractCLRoutine):

    def __init__(self, cl_environments, load_balancer, double_precision):
        """Given a set of raw errors per voxel, calculate some interesting measures."""
        super(ErrorMeasures, self).__init__(cl_environments, load_balancer)
        self._double_precision = double_precision

    def calculate(self, errors):
        """Given a set of raw errors per voxel, calculate some interesting measures.

        Args:
            errors (ndarray): The list with errors per problem instance.

        Returns:
            A dictionary containing (for each voxel)
                - Errors.l2: the l2 norm
        """
        return {'Errors.l2': np.linalg.norm(errors, axis=1)}