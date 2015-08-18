from scipy.optimize import basinhopping, fmin_bfgs, leastsq, minimize, fmin_powell
from .base import AbstractSerialOptimizer
from ...cl_python_callbacks import CLToPythonCallbacks

__author__ = 'Robbert Harms'
__date__ = "2015-04-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SerialBasinHopping(AbstractSerialOptimizer):

    def __init__(self, cl_environments, load_balancer, use_param_codec=True, patience=None,
                 basinhopping_args=None):
        """Uses basinhopping from scipy.optimize"""
        super(SerialBasinHopping, self).__init__(cl_environments, load_balancer, use_param_codec)
        self.basinhopping_args = basinhopping_args or {'method': 'powell'}

    def _minimize_single_voxel(self, objective_cb, x0):
        return basinhopping(objective_cb, x0, minimizer_kwargs=self.basinhopping_args).x


class SerialBFGS(AbstractSerialOptimizer):

    def __init__(self, cl_environments, load_balancer, use_param_codec=True, patience=None):
        """Uses fmin_bfgs from scipy.optimize"""
        super(SerialBFGS, self).__init__(cl_environments, load_balancer, use_param_codec)

    def _minimize_single_voxel(self, objective_cb, x0):
        return fmin_bfgs(objective_cb, x0, disp=False)


class SerialLM(AbstractSerialOptimizer):

    def __init__(self, cl_environments, load_balancer, use_param_codec=True, patience=None):
        """Uses leastsq from scipy.optimize"""
        super(SerialLM, self).__init__(cl_environments, load_balancer, use_param_codec)

    def _minimize(self, model, starting_points, cl_environment):
        cb_generator = CLToPythonCallbacks(model, cl_environment=cl_environment)

        for voxel_index in range(model.get_nmr_problems()):
            residual_cb = cb_generator.get_residual_cb(voxel_index, decode_params=True)
            x0 = starting_points[voxel_index, :]
            x_opt = leastsq(residual_cb, x0)[0]
            starting_points[voxel_index, :] = x_opt

        return starting_points


class SerialNMSimplex(AbstractSerialOptimizer):

    def __init__(self, cl_environments, load_balancer, use_param_codec=True, patience=None):
        """Uses minimize from scipy.optimize"""
        super(SerialNMSimplex, self).__init__(cl_environments, load_balancer, use_param_codec)

    def _minimize_single_voxel(self, objective_cb, x0):
        return minimize(objective_cb, x0, method='nelder-mead', options={'disp': False}).x


class SerialPowell(AbstractSerialOptimizer):

    def __init__(self, cl_environments, load_balancer, use_param_codec=True, patience=None):
        """Uses fmin_powell from scipy.optimize"""
        super(SerialPowell, self).__init__(cl_environments, load_balancer, use_param_codec)

    def _minimize_single_voxel(self, objective_cb, x0):
        return fmin_powell(objective_cb, x0, disp=False)