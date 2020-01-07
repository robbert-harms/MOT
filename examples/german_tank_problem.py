import numpy as np
import matplotlib.pyplot as plt
from mot.lib.cl_function import SimpleCLFunction
from mot.random import normal, uniform
from mot.sample import AdaptiveMetropolisWithinGibbs
from mot.lib.kernel_data import Array, Struct

__author__ = 'Robbert Harms'
__date__ = '2018-04-04'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def get_historical_data(nmr_problems):
    """Get the historical tank data.

    Args:
        nmr_problems (int): the number of problems

    Returns:
        tuple: (observations, nmr_tanks_ground_truth)
    """
    observations = np.tile(np.array([[10, 256, 202, 97]]), (nmr_problems, 1))
    nmr_tanks_ground_truth = np.ones((nmr_problems,)) * 276
    return observations, nmr_tanks_ground_truth


def get_simulated_data(nmr_problems):
    """Simulate some data.

    This returns the simulated tank observations and the corresponding ground truth maximum number of tanks.

    Args:
        nmr_problems (int): the number of problems

    Returns:
        tuple: (observations, nmr_tanks_ground_truth)
    """
    # The number of tanks we observe per problem
    nmr_observed_tanks = 10

    # Generate some maximum number of tanks. Basically the ground truth of the estimation problem.
    nmr_tanks_ground_truth = normal(nmr_problems, 1, mean=250, std=30, ctype='uint')

    # Generate some random tank observations
    observations = uniform(nmr_problems, nmr_observed_tanks, low=0, high=nmr_tanks_ground_truth, ctype='uint')

    return observations, nmr_tanks_ground_truth


def get_log_likelihood_function(nmr_observed_tanks):
    return SimpleCLFunction.from_string('''
        double germanTank_logLikelihood(const mot_float_type* const x, void* data){

            uint nmr_tanks = (uint)round(x[0]);
            double sum = 0;
            double eval;
            for(uint i = 0; i < ''' + str(nmr_observed_tanks) + ''' - 1; i++){
                eval = discrete_uniform(((_model_data*)data)->observed_tanks[i], 1, nmr_tanks);
                sum += eval;
            }
            return sum;
        }
        ''', dependencies=[discrete_uniform_func()])


def get_log_prior_function():
    return SimpleCLFunction.from_string('''
        double germanTank_logPrior(const mot_float_type* const x, void* data){
            uint nmr_tanks = (uint)round(x[0]);
            return discrete_uniform(
                nmr_tanks, ((_model_data*)data)->lower_bounds[0], ((_model_data*)data)->upper_bounds[0]);
        }
    ''', dependencies=[discrete_uniform_func()])


def discrete_uniform_func():
    return SimpleCLFunction.from_string('''
        float discrete_uniform(uint x, uint lower, uint upper){
            if(x < lower || x > upper){
                return -INFINITY;
            }
            return -log((float)(upper-lower));
        }
    ''')


if __name__ == '__main__':
    """Runs the German Tank problem.

    The German Tank problem is an univariate parameter estimation problem in which we try to get a best estimate on
    the number of manufactured tanks, given that we observed some serial numbers on some tanks.

    For more information, see http://matatat.org/sampyl/examples/german_tank_problem.html or
    https://en.wikipedia.org/wiki/German_tank_problem
    """

    # The number of problems
    nmr_problems = 10000

    # The data we would like to use
    observations, nmr_tanks_ground_truth = get_historical_data(nmr_problems)
    # observations, nmr_tanks_ground_truth = get_simulated_data(nmr_problems)


    ## Sample ##
    # The additional data we need
    kernel_data = Struct({'observed_tanks': Array(observations, 'uint'),
                          'lower_bounds': Array(np.max(observations, axis=1), 'uint'),
                          'upper_bounds': Array(np.ones((nmr_problems,)) * 1000, 'uint')}, '_model_data')

    # Create an instance of the sample routine we want to use.
    sampler = AdaptiveMetropolisWithinGibbs(
        get_log_likelihood_function(observations.shape[1]),
        get_log_prior_function(),
        np.max(observations, axis=1),  # starting position
        np.ones(nmr_problems) * 10,  # initial proposal standard deviations
        data=kernel_data)

    # Sample each instance
    sampling_output = sampler.sample(10000, thinning=1, burnin=0)

    # Obtain the samples
    samples = sampling_output.get_samples()

    # Histogram of for the first 5 chains
    for problem_instance in range(min(nmr_problems, 5)):
        param_ind = 0
        posterior = np.round(samples[problem_instance, param_ind])

        print('Problem instance, estimate mean, ground truth: ',
              problem_instance, np.mean(posterior), np.squeeze(nmr_tanks_ground_truth[problem_instance]))

        plt.hist(posterior, bins=500)
        plt.xlabel("Total number of tanks")
        plt.ylabel("Posterior probability mass")
        plt.show()

    # Histogram of the errors
    errors = np.squeeze(nmr_tanks_ground_truth) - np.squeeze(np.mean(samples, axis=2))

    print('Mean absolute errors:', np.mean(np.abs(errors)))

    plt.hist(errors)
    plt.xlabel("Mean absolute error")
    plt.ylabel("Probability mass")
    plt.show()

