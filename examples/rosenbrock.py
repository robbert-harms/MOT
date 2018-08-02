import numpy as np
import matplotlib.pyplot as plt
from mot.optimize import minimize
from mot.lib.cl_function import SimpleCLFunction
from mot.sample import AdaptiveMetropolisWithinGibbs
from mot.lib.model_interfaces import SampleModelInterface, OptimizeModelInterface

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
            nmr_problems (int): the number of parallel optimization or sample chains
            nmr_params (int): the number of Rosenbrock parameters
        """
        super(SampleModelInterface, self).__init__()
        self.nmr_problems = nmr_problems
        self.nmr_params = nmr_params


    ## Methods used for both optimization and sample ##
    def get_kernel_data(self):
        return {}

    def get_nmr_observations(self):
        return self.nmr_params


    ## Methods used for optimization ##
    def get_objective_function(self):
        """Used in Maximum Likelihood Estimation."""
        return SimpleCLFunction.from_string('''
            double rosenbrock_MLE_func(mot_data_struct* data, 
                                       local const mot_float_type* const x,
                                       local mot_float_type* objective_list,
                                       local double* objective_value_tmp){
                
                double sum = 0;
                double eval;
                for(uint i = 0; i < ''' + str(self.get_nmr_observations()) + ''' - 1; i++){
                    eval = 100 * pown(x[i + 1] - pown(x[i], 2), 2) + pown(1 - x[i], 2);
                    sum += eval;
                    
                    if(objective_list){
                        objective_list[i] = eval;
                    }
                }
                return sum;
            }
        ''')

    def get_lower_bounds(self):
        return [-np.inf] * self.nmr_params

    def get_upper_bounds(self):
        return [np.inf] * self.nmr_params


    ## Methods used for sample ##
    def get_log_likelihood_function(self):
        """Used in Bayesian sample."""
        return SimpleCLFunction.from_string('''
            double rosenbrock_logLikelihood(
                    mot_data_struct* data, 
                    local const mot_float_type* const x, 
                    local double* objective_value_tmp){
                
                double sum = 0;
                double eval;
                for(uint i = 0; i < ''' + str(self.get_nmr_observations()) + ''' - 1; i++){
                    eval = -(100 * pown(x[i + 1] - pown(x[i], 2), 2) + pown(1 - x[i], 2));
                    sum += eval;
                }
                return sum;
            }
        ''')

    def get_log_prior_function(self):
        """Used in Bayesian sample."""
        return SimpleCLFunction.from_string('''
            double rosenbrock_logPrior(mot_data_struct* data, local const mot_float_type* const x){
                for(uint i = 0; i < ''' + str(self.nmr_params) + '''; i++){
                    if(x[i] < -10 || x[i] > 10){
                        return log(0.0);
                    }
                }
                return log(1.0);
            }
        ''')


if __name__ == '__main__':
    # How many Rosenbrock dimensions/parameters we want to fit and sample
    nmr_params = 2

    # How many times we want to solve the Rosenbrock function
    nmr_problems = 10000

    # Create the Rosenbrock model for the given number of problem instances and parameters
    model = Rosenbrock(nmr_problems, nmr_params)


    ## Optimization ##
    # The optimization starting points
    x0 = np.ones((nmr_problems, nmr_params)) * 3

    # Minimize the parameters of the model given the starting points.
    opt_output = minimize(model, x0, options={'patience': 5})

    # Print the output
    print(opt_output['x'])


    ## Sampling ##
    # The initial proposal standard deviations
    proposal_stds = np.ones_like(x0)

    # Create an instance of the sample routine we want to use.
    sampler = AdaptiveMetropolisWithinGibbs(model, x0, proposal_stds)

    # Sample each Rosenbrock instance
    sampling_output = sampler.sample(10000, thinning=1, burnin=0)

    # Obtain the samples
    samples = sampling_output.get_samples()

    # Scatter plot of the first two dimensions, for the first 5 problems
    for problem_instance in range(min(nmr_problems, 5)):
        plt.scatter(samples[problem_instance, 0], samples[problem_instance, 1])
        plt.show()
