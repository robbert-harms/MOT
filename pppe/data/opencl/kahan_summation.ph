#ifndef KAHANSUM_%(MEMSPACE)s_H
#define KAHANSUM_%(MEMSPACE)s_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-01
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/*
 * Implementation of the Kahan summation algorithm, taken from the
 * following Wikipedia entry:
 * http://en.wikipedia.org/wiki/Kahan_summation_algorithm
 */
double kahan_summation_%(MEMSPACE)s(const %(MEMSPACE)s double* const l, const int n);

#endif // KAHANSUM_%(MEMSPACE)s_H