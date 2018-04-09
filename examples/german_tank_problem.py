import numpy as np
import matplotlib.pyplot as plt
from mot import Powell
from mot.cl_routines.sampling.amwg import AdaptiveMetropolisWithinGibbs
from mot.model_interfaces import SampleModelInterface
from mot.utils import SimpleNamedCLFunction, KernelInputArray

__author__ = 'Robbert Harms'
__date__ = '2018-04-04'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class GermanTanks(SampleModelInterface):

    def __init__(self, observed_tanks, upper_bounds=10000):
        """Create the MOT model for the German Tank problem.

        The German Tank problem is an univariate parameter estimation problem in which we try to estimate the
        number of tanks that were made given that we observed some serial numbers on some tanks.

        For more information, see http://matatat.org/sampyl/examples/german_tank_problem.html or
        https://en.wikipedia.org/wiki/German_tank_problem

        Args:
            observed_tanks (ndarray): a two dimensional array as (nmr_problems, nmr_tanks). That is, on the
                first dimension the number of problem instances (nmr of unique optimization or sampling instances)
                and on the second dimension the number of observed tanks. Each element in the matrix is an integer
                with the observed tank number.
            upper_bounds (int or ndarray): the upper bounds for the computations. Can be one upper bound for all
                problems, or an upper bound per problem.
        """
        super(SampleModelInterface, self).__init__()
        self.observed_tanks = observed_tanks
        self.upper_bounds = upper_bounds

        # todo upper bounds to matrix

    def get_kernel_data(self):
        return {'observed_tanks': KernelInputArray(self.observed_tanks, 'uint'),
                'max_tank_number': KernelInputArray(np.max(self.observed_tanks, axis=1), 'uint')}

    def get_nmr_problems(self):
        return self.observed_tanks.shape[0]

    def get_nmr_inst_per_problem(self):
        return self.observed_tanks.shape[1]

    def get_nmr_estimable_parameters(self):
        return 1

    def get_objective_per_observation_function(self):
        func_name = 'getObjectiveInstanceValue'
        func = '''
            mot_float_type ''' + func_name + '''(void* data, const mot_float_type* const x, uint observation_index){
                if(x[0] < data->max_tank_number || x[0] >  
                
            }
        '''
        return SimpleNamedCLFunction(func, func_name)

    def get_lower_bounds(self):
        return np.max(self.observed_tanks, axis=-1)

    def get_upper_bounds(self):
        return self.upper_bounds

    def get_log_likelihood_per_observation_function(self):
        """Needed for sampling"""
        fname = 'logLikelihood'
        func = '''
            double ''' + fname + '''(mot_data_struct* data, const mot_float_type* const x, uint observation_index){
                uint i = observation_index;
                return -(100 * pown(x[i + 1] - pown(x[i], 2), 2) + pown(1 - x[i], 2));
            }
        '''
        return SimpleNamedCLFunction(func, fname)

    def get_log_prior_function(self, address_space_parameter_vector='private'):
        """Needed for sampling"""
        fname = 'logPrior'
        func = '''
            double ''' + fname + '''(mot_data_struct* data,
                    ''' + str(address_space_parameter_vector) + ''' const mot_float_type* const x){
                
                for(uint i = 0; i < ''' + str(self.nmr_params) + '''; i++){
                    if(x[i] < -10 || x[i] > 10){
                        return log(0.0);
                    }
                }
                return log(1.0);
            }
        '''
        return SimpleNamedCLFunction(func, fname)


if __name__ == '__main__':
    # How many Rosenbrock dimensions we want
    nmr_params = 2

    # How many times we want to solve the Rosenbrock function
    nmr_problems = 10000

    # Create the Rosenbrock model for the given number of problem instances and parameters
    model = GermanTanks(nmr_problems, nmr_params)

    # Create an instance of the optimization routine we will use
    optimizer = Powell(patience=5)

    # The optimization starting points
    starting_points = np.ones((nmr_problems, nmr_params)) * 3

    # Minimize the parameters of the model given the starting points.
    opt_output = optimizer.minimize(model, starting_points)

    # Print the output
    print(opt_output.get_optimization_result())

    # The initial proposal standard deviations
    proposal_stds = np.ones_like(starting_points)

    # Create an instance of the sampling routine we want to use.
    sampler = AdaptiveMetropolisWithinGibbs(model, starting_points, proposal_stds)

    # Sample each Rosenbrock instance
    sampling_output = sampler.sample(10000, thinning=1, burnin=0)

    # Obtain the samples
    samples = sampling_output.get_samples()

    # Scatter plot of the first two dimensions, for the first 5 problems
    for problem_instance in range(max(nmr_problems, 5)):
        plt.scatter(samples[problem_instance, 0], samples[problem_instance, 1])
        plt.show()
