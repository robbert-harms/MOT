import numpy as np
import matplotlib.pyplot as plt
from mot.optimize import minimize
from mot.lib.cl_function import SimpleCLFunction
from mot.sample import AdaptiveMetropolisWithinGibbs

__author__ = 'Robbert Harms'
__date__ = '2018-04-04'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'


def get_objective_function(nmr_parameters):
    return SimpleCLFunction.from_string('''
        double rosenbrock_MLE_func(const mot_float_type* const x,
                                   void* data,
                                   mot_float_type* objective_list){

            double sum = 0;
            double eval;
            for(uint i = 0; i < ''' + str(nmr_parameters) + ''' - 1; i++){
                eval = 100 * pown(x[i + 1] - pown(x[i], 2), 2) + pown(1 - x[i], 2);
                sum += eval;

                if(objective_list){
                    objective_list[i] = eval;
                }
            }
            return sum;
        }
    ''')


def get_log_likelihood_function(nmr_parameters):
    return SimpleCLFunction.from_string('''
        double rosenbrock_logLikelihood(const mot_float_type* const x, void* data){
            double sum = 0;
            double eval;
            for(uint i = 0; i < ''' + str(nmr_parameters) + ''' - 1; i++){
                eval = -(100 * pown(x[i + 1] - pown(x[i], 2), 2) + pown(1 - x[i], 2));
                sum += eval;
            }
            return sum;
        }
    ''')


def get_log_prior_function(nmr_parameters):
    return SimpleCLFunction.from_string('''
        double rosenbrock_logPrior(const mot_float_type* const x, void* data){
            for(uint i = 0; i < ''' + str(nmr_parameters) + '''; i++){
                if(fabs(x[i]) > 10){
                    return log(0.0);
                }
            }
            return log(1.0);
        }
    ''')


if __name__ == '__main__':
    """MOT example of fitting the multidimensional generalized Rosenbrock function.

    This creates ``nmr_problems`` Rosenbrock instances, each with ``nmr_params`` parameters.

    The Rosenbrock function we use here is defined as (https://en.wikipedia.org/wiki/Rosenbrock_function):

    .. math::

        f(\mathbf{x}) = \sum_{i=0}^{N-2} 100(x_{i+1}-x_{i}^{2})^{2}+(1-x_{i})^{2}\quad {\mbox{where}}
        \quad \mathbf {x} =[x_{0},\ldots ,x_{N-1}]\in \mathbb {R} ^{N}.


    When optimized the parameters should all be equal to 1.
    """

    # How many Rosenbrock dimensions/parameters we want to fit and sample
    nmr_params = 2

    # How many unique instances of the Rosenbrock function
    nmr_problems = 10000


    ## Optimization ##
    # The optimization starting points
    x0 = np.ones((nmr_problems, nmr_params)) * 3

    # Minimize the parameters of the model given the starting points.
    opt_output = minimize(get_objective_function(nmr_params), x0, options={'patience': 5})

    # Print the output
    print(opt_output['x'])


    ## Sampling ##
    # Create an instance of the sample routine we want to use.
    sampler = AdaptiveMetropolisWithinGibbs(
        get_log_likelihood_function(nmr_params),
        get_log_prior_function(nmr_params),
        x0,
        np.ones_like(x0)  # The initial proposal standard deviations
    )

    # Sample each Rosenbrock instance
    sampling_output = sampler.sample(10000, thinning=1, burnin=0)

    # Obtain the samples
    samples = sampling_output.get_samples()

    # Scatter plot of the first two dimensions, for the first 5 problems
    for problem_instance in range(min(nmr_problems, 5)):
        plt.scatter(samples[problem_instance, 0], samples[problem_instance, 1])
        plt.show()
