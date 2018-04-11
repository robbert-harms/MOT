import numpy as np
import matplotlib.pyplot as plt
from mot import Powell
from mot.cl_routines.sampling.amwg import AdaptiveMetropolisWithinGibbs
from mot.model_interfaces import SampleModelInterface, OptimizeModelInterface
from mot.utils import NameFunctionTuple

__author__ = 'Robbert Harms'
__date__ = '2018-04-04'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class Rosenbrock(OptimizeModelInterface, SampleModelInterface):

    def __init__(self, nmr_problems, nmr_params):
        """MOT model definition of the multidimensional generalized Rosenbrock function.

        This creates ``nmr_problems`` Rosenbrock instances, each with ``nmr_params`` parameters.

        The Rosenbrock function we use here is defined as (https://en.wikipedia.org/wiki/Rosenbrock_function):

        .. math::

            f(\mathbf{x}) = \sum_{i=0}^{N-2} 100(x_{i+1}-x_{i}^{2})^{2}+(1-x_{i})^{2}\quad {\mbox{where}}
            \quad \mathbf {x} =[x_{0},\ldots ,x_{N-1}]\in \mathbb {R} ^{N}.


        When optimized the parameters should all be equal to 1.

        Args:
            nmr_problems (int): the number of parallel optimization or sampling chains
            nmr_params (int): the number of Rosenbrock parameters
        """
        super(SampleModelInterface, self).__init__()
        self.nmr_problems = nmr_problems
        self.nmr_params = nmr_params


    ## Methods used for both optimization and sampling ##
    def get_kernel_data(self):
        return {}

    def get_nmr_problems(self):
        return self.nmr_problems

    def get_nmr_inst_per_problem(self):
        return self.nmr_params

    def get_nmr_parameters(self):
        return self.nmr_params


    ## Methods used for optimization ##
    def get_objective_per_observation_function(self):
        """Used in Maximum Likelihood Estimation."""
        func_name = 'rosenbrock_MLE_func'
        func = '''
            mot_float_type ''' + func_name + '''(mot_data_struct* data, const mot_float_type* const x, 
                                                 uint observation_index){
                uint i = observation_index;
                return 100 * pown(x[i + 1] - pown(x[i], 2), 2) + pown(1 - x[i], 2);
            }
        '''
        return NameFunctionTuple(func_name, func)

    def get_lower_bounds(self):
        return [-np.inf] * self.nmr_params

    def get_upper_bounds(self):
        return [np.inf] * self.nmr_params


    ## Methods used for sampling ##
    def get_log_likelihood_per_observation_function(self):
        """Used in Bayesian sampling."""
        fname = 'rosenbrock_logLikelihood'
        func = '''
            double ''' + fname + '''(mot_data_struct* data, const mot_float_type* const x, 
                                     uint observation_index){
                uint i = observation_index;
                return -(100 * pown(x[i + 1] - pown(x[i], 2), 2) + pown(1 - x[i], 2));
            }
        '''
        return NameFunctionTuple(fname, func)

    def get_log_prior_function(self, address_space_parameter_vector='private'):
        """Used in Bayesian sampling."""
        fname = 'rosenbrock_logPrior'
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
        return NameFunctionTuple(fname, func)


if __name__ == '__main__':
    # How many Rosenbrock dimensions/parameters we want to fit and sample
    nmr_params = 2

    # How many times we want to solve the Rosenbrock function
    nmr_problems = 10000

    # Create the Rosenbrock model for the given number of problem instances and parameters
    model = Rosenbrock(nmr_problems, nmr_params)


    ## Optimization ##
    # Create an instance of the optimization routine we will use
    optimizer = Powell(patience=5)

    # The optimization starting points
    starting_points = np.ones((nmr_problems, nmr_params)) * 3

    # Minimize the parameters of the model given the starting points.
    opt_output = optimizer.minimize(model, starting_points)

    # Print the output
    print(opt_output.get_optimization_result())


    ## Sampling ##
    # The initial proposal standard deviations
    proposal_stds = np.ones_like(starting_points)

    # Create an instance of the sampling routine we want to use.
    sampler = AdaptiveMetropolisWithinGibbs(model, starting_points, proposal_stds)

    # Sample each Rosenbrock instance
    sampling_output = sampler.sample(10000, thinning=1, burnin=0)

    # Obtain the samples
    samples = sampling_output.get_samples()

    # Scatter plot of the first two dimensions, for the first 5 problems
    for problem_instance in range(min(nmr_problems, 5)):
        plt.scatter(samples[problem_instance, 0], samples[problem_instance, 1])
        plt.show()
