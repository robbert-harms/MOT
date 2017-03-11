/**
 * Author = Robbert Harms
 * Date = 2017-03-11
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Computes log(cosh(x)).
 *
 * For large x this will try to estimate it without overflow. For small x we use the opencl functions log and cos.
 *
 * The estimation for large numbers has been taken from:
 * https://github.com/JaneliaSciComp/tmt/blob/master/basics/logcosh.m
 *
 */
double log_cosh(double x){
   if(x < 50){
        return log(cosh(x));
    }
    return fabs(x) + log(1 + exp(-2.0 * fabs(x))) - log(2.0);
}
