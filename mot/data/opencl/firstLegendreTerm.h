#ifndef FIRST_LEGENDRE_TERM_H
#define FIRST_LEGENDRE_TERM_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-01
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
* Compute the first term of the legendre polynome for the given value x and the polynomial degree n
*/
model_float getFirstLegendreTerm(const model_float x, const int n);

#endif // FIRST_LEGENDRE_TERM_H