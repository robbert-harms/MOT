import numpy as np
import matplotlib.pyplot as plt

from mot.cl_function import SimpleCLFunction
from mot.cl_routines.generate_random import randn, rand
from mot.cl_routines.sampling.amwg import AdaptiveMetropolisWithinGibbs
from mot.model_interfaces import SampleModelInterface
from mot.kernel_data import Array

__author__ = 'Robbert Harms'
__date__ = '2018-04-04'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class GermanTanks(SampleModelInterface):

    def __init__(self, observed_tanks, upper_bounds):
        """Create the MOT model for the German Tank problem.

        The German Tank problem is an univariate parameter estimation problem in which we try to get a best estimate on
        the number of manufactured tanks, given that we observed some serial numbers on some tanks.

        For more information, see http://matatat.org/sampyl/examples/german_tank_problem.html or
        https://en.wikipedia.org/wiki/German_tank_problem

        Args:
            observed_tanks (ndarray): a two dimensional array as (nmr_problems, nmr_tanks). That is, on the
                first dimension the number of problem instances (nmr of unique optimization or sampling instances)
                and on the second dimension the number of observed tanks. Each element is an integer
                with an observed tank number.
            upper_bounds (ndarray): per problem an estimated upper bound for the number of tanks
        """
        super(SampleModelInterface, self).__init__()
        self.observed_tanks = observed_tanks
        self.upper_bounds = upper_bounds

    def get_kernel_data(self):
        return {'observed_tanks': Array(self.observed_tanks, 'uint'),
                'lower_bounds': Array(np.max(self.observed_tanks, axis=1), 'uint'),
                'upper_bounds': Array(self.upper_bounds, 'uint')}

    def get_nmr_observations(self):
        return self.observed_tanks.shape[1]

    def get_log_likelihood_function(self):
        """Used in Bayesian sampling."""
        return SimpleCLFunction.from_string('''
            double germanTank_logLikelihood(mot_data_struct* data, 
                                            local const mot_float_type* const x,
                                            local double* objective_value_tmp){
                
                uint nmr_tanks = (uint)round(x[0]);
                double sum = 0;
                double eval;
                for(uint i = 0; i < ''' + str(self.get_nmr_observations()) + ''' - 1; i++){
                    eval = discrete_uniform(data->observed_tanks[i], 1, nmr_tanks);
                    sum += eval;
                }
                return sum;   
            }
            ''', dependencies=[self._discrete_uniform()])

    def get_log_prior_function(self):
        """Used in Bayesian sampling."""
        return SimpleCLFunction.from_string('''
            double germanTank_logPrior(mot_data_struct* data, local const mot_float_type* const x){
                uint nmr_tanks = (uint)round(x[0]);
                return discrete_uniform(nmr_tanks, data->lower_bounds[0], data->upper_bounds[0]);
            }
        ''', dependencies=[self._discrete_uniform()])

    def _discrete_uniform(self):
        return SimpleCLFunction.from_string('''
            float discrete_uniform(uint x, uint lower, uint upper){
                if(x < lower || x > upper){
                    return -INFINITY;
                }
                return -log((float)(upper-lower));
            }
        ''')


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
    nmr_tanks_ground_truth = randn(nmr_problems, 1, mean=250, std=30, ctype='uint')

    # Generate some random tank observations
    observations = rand(nmr_problems, nmr_observed_tanks, min_val=0,
                        max_val=nmr_tanks_ground_truth, ctype='uint')

    return observations, nmr_tanks_ground_truth


if __name__ == '__main__':
    # The number of problems
    nmr_problems = 10000

    # The data we would like to use
    # observations, nmr_tanks_ground_truth = get_simulated_data(nmr_problems)
    observations, nmr_tanks_ground_truth = get_historical_data(nmr_problems)

    ## Sample ##
    # Estimation upper bound
    upper_bounds = np.ones((nmr_problems,)) * 1000

    # Create the model with the observations and upper bounds
    model = GermanTanks(observations, upper_bounds)

    # The starting points
    starting_points = np.max(observations, axis=1)

    # The initial proposal standard deviations
    proposal_stds = np.ones_like(starting_points) * 10

    # Create an instance of the sampling routine we want to use.
    sampler = AdaptiveMetropolisWithinGibbs(model, starting_points, proposal_stds)

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

