#ifndef MCMC_STRETCH_H
#define MCMC_STRETCH_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-01
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Required compile time definitions.
 *
 * #define NMR_PARAMS                     | as the number of parameters to fit (n), the dimension of the program
 * #define EVAL_FUNC_NAME                 | the name of the eval function the routine should call
 * #define K_OVER_TWO                     | Number of walkers in each group
 * #define A_COEFF_0                      | the first component of the a parameter
 * #define A_COEFF_1                      | the second component of the a parameter
 * #define A_COEFF_2                      | the third component of the a parameter
 */

void mcmc_stretch(
    __global float *X_moving,                  // walkers to be updated
    __global float *log_prob_moving,           // cached log probabilities of the moving walkers, will be updated
    __global const float *X_fixed,             // fixed walkers
    __global float4 *ranluxcltab,              // state information for random number generator
    __global unsigned long *accepted,          // number of samples accepted
    const void *data,                          // data or observations
    const float beta);


#endif // MCMC_STRETCH_H