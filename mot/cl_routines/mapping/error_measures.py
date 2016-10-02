import numpy as np
from ...cl_routines.base import CLRoutine


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ErrorMeasures(CLRoutine):

    def __init__(self, cl_environments=None, load_balancer=None, double_precision=False):
        """Given a set of raw errors per voxel, calculate some interesting measures."""
        super(ErrorMeasures, self).__init__(cl_environments=cl_environments, load_balancer=load_balancer)
        self._double_precision = double_precision

    def calculate(self, errors):
        """Given a set of raw errors per voxel, calculate some interesting measures.

        Args:
            errors (ndarray): The list with errors per problem instance.

        Returns:
            A dictionary containing (for each voxel)
                - Errors.l2: the l2 norm (square root of sum of squares)
                - Errors.mse: the mean sum of squared errors
        """
        sse = np.sum(np.power(errors, 2), axis=1)
        return {'Errors.l2': np.linalg.norm(errors, axis=1),
                'Errors.mse': sse/errors.shape[1]}
