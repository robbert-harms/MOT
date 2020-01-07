import numpy as np

from mot.configuration import CLRuntimeInfo
from mot.lib.kernel_data import Array
from mot.library_functions import gamma_logpdf
from mot.optimize import minimize
from mot.lib.cl_function import SimpleCLFunction


__author__ = 'Robbert Harms'
__date__ = '2018-04-04'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def get_objective_function(nmr_datapoints):
    return SimpleCLFunction.from_string('''
        double fit_gamma_distribution(const mot_float_type* const x,
                                      void* data,
                                      mot_float_type* objective_list){

            if(x[0] < 0 || x[1] < 0){
                return INFINITY;
            }

            double sum = 0;

            uint batch_range;
            uint offset = get_workitem_batch(''' + str(nmr_datapoints) + ''', &batch_range);
            for(uint i = offset; i < offset + batch_range; i++){
                sum += gamma_logpdf(((float*)data)[i] , x[0], x[1]);
            }
            return -work_group_reduce_add(sum); // the optimization routines are minimizers
        }
    ''', dependencies=[gamma_logpdf()])


if __name__ == '__main__':
    """MOT example of estimating the shape and scale parameters of the gamma distribution, given some data.

    This first simulates some test data, with ``nmr_simulations`` as the number of unique simulations and
    ``nmr_datapoints`` as the number of data points per simulation.

    Since we generate only 25 random datapoints on the simulated Gamma distribution, the fitting results may not
    be perfect for every simulated distribution. In general though, fit results should match the ground truth.
    """
    # The number of unique distributions, this is typically very large
    nmr_simulations = 2

    # How many data points per distribution, this is typically small
    nmr_datapoints = 25

    # generate a range of parameters, basically the ground truth
    shape = np.random.uniform(0.1, 10, nmr_simulations)
    scale = np.random.uniform(0.1, 5, nmr_simulations)

    # generate some random locations on those simulated distributions
    gamma_random = np.zeros((nmr_simulations, nmr_datapoints))
    for i in range(nmr_datapoints):
        gamma_random[:, i] = np.random.gamma(shape, scale)

    # The optimization starting points for shape and scale
    x0 = np.ones((nmr_simulations, 2))

    # Minimize the parameters of the model given the starting points.
    opt_output = minimize(get_objective_function(nmr_datapoints),
                          x0,
                          data=Array(gamma_random, ctype='float'),
                          cl_runtime_info=CLRuntimeInfo(cl_environments=[0, 1]), use_local_reduction=False)

    # Print the output
    # print(np.column_stack([shape, scale]))
    print(opt_output.x)
    # print(np.abs(opt_output.x - np.column_stack([shape, scale])))
