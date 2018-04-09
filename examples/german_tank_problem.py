import numpy as np
import matplotlib.pyplot as plt
from mot import Powell
from mot.cl_routines.sampling.amwg import AdaptiveMetropolisWithinGibbs
from mot.cl_routines.sampling.scam import SingleComponentAdaptiveMetropolis
from mot.model_interfaces import SampleModelInterface
from mot.utils import SimpleNamedCLFunction, KernelInputArray, add_include_guards

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
        return {'observed_tanks': KernelInputArray(self.observed_tanks, 'uint'),
                'lower_bounds': KernelInputArray(np.max(self.observed_tanks, axis=1), 'uint'),
                'upper_bounds': KernelInputArray(self.upper_bounds, 'uint')}

    def get_nmr_problems(self):
        return self.observed_tanks.shape[0]

    def get_nmr_inst_per_problem(self):
        return self.observed_tanks.shape[1]

    def get_nmr_parameters(self):
        return 1

    def get_objective_per_observation_function(self):
        """Returns, per observation, the negative of the Discrete Uniform distribution's log-likelihood.
        """
        func_name = 'germanTank_neglogLikelihood'

        func = self._discrete_uniform()
        func += '''
            mot_float_type ''' + func_name + '''(mot_data_struct* data, const mot_float_type* const x, 
                                                 uint observation_index){
                uint nmr_tanks = (uint)round(x[0]);
                return -discrete_uniform(data->observed_tanks[observation_index], 1, nmr_tanks); 
            }
        '''
        return SimpleNamedCLFunction(func, func_name)

    def get_lower_bounds(self):
        return np.max(self.observed_tanks, axis=1)

    def get_upper_bounds(self):
        return self.upper_bounds

    def _discrete_uniform(self):
        return add_include_guards('''
            float discrete_uniform(uint x, uint lower, uint upper){
                if(x < lower || x > upper){
                    return -INFINITY;
                }
                return -log((float)(upper-lower));
            }
        ''')

    def get_log_likelihood_per_observation_function(self):
        """Used in Bayesian sampling."""
        fname = 'germanTank_logLikelihood'

        func = self._discrete_uniform()
        func += '''
            double ''' + fname + '''(mot_data_struct* data, const mot_float_type* const x, 
                                     uint observation_index){
                uint nmr_tanks = (uint)round(x[0]);
                
                printf("%i, %i, %f", data->observed_tanks[observation_index], nmr_tanks, discrete_uniform(data->observed_tanks[observation_index], 1, nmr_tanks));
                
                return discrete_uniform(data->observed_tanks[observation_index], 1, nmr_tanks); 
            }
        '''
        return SimpleNamedCLFunction(func, fname)

    def get_log_prior_function(self, address_space_parameter_vector='private'):
        """Used in Bayesian sampling."""
        fname = 'germanTank_logPrior'

        func = self._discrete_uniform()
        func += '''
            double ''' + fname + '''(mot_data_struct* data,
                    ''' + str(address_space_parameter_vector) + ''' const mot_float_type* const x){
                
                uint nmr_tanks = (uint)round(x[0]);
                return discrete_uniform(nmr_tanks, data->lower_bounds, data->upper_bounds);
            }
        '''
        return SimpleNamedCLFunction(func, fname)


if __name__ == '__main__':
    observations = np.array([[10, 256, 202, 97.]])
    upper_bounds = np.array([10000])

    model = GermanTanks(observations, upper_bounds)

    # The starting points
    starting_points = np.max(observations, axis=1) + 100

    # ## Optimization ##
    # # Create an instance of the optimization routine we will use
    # optimizer = Powell(patience=5)
    #
    #
    # # Minimize the parameters of the model given the starting points.
    # opt_output = optimizer.minimize(model, starting_points)
    #
    # # Print the output
    # print(opt_output.get_optimization_result())


    ## Sampling ##
    # The initial proposal standard deviations
    proposal_stds = np.ones_like(starting_points) * 10

    # Create an instance of the sampling routine we want to use.
    sampler = AdaptiveMetropolisWithinGibbs(model, starting_points, proposal_stds)

    # Sample each instance
    sampling_output = sampler.sample(10, thinning=1, burnin=0)

    # Obtain the samples
    samples = sampling_output.get_samples()

    plt.hist(samples[0])
    plt.show()

    # # Scatter plot of the first two dimensions, for the first 5 problems
    # for problem_instance in range(max(nmr_problems, 5)):
    #     plt.scatter(samples[problem_instance, 0], samples[problem_instance, 1])
    #     plt.show()
#
