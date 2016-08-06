#ifndef FIRST_LEGENDRE_TERM_CL
#define FIRST_LEGENDRE_TERM_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-01
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

////////////////////////////////////////////////////////////////////////////////
// double getFirstLegendreTerm(double x, int n)                               //
//                                                                            //
//  Description:                                                              //
//     The Legendre polynomials, Pn(x), are orthogonal on the interval [-1,1] //
//     with weight function w(x) = 1 for -1 <= x <= 1 and 0 elsewhere.  They  //
//     are normalized so that Pn(1) = 1.  The inner products are:             //
//             <Pn,Pm> = 0        if n != m,                                  //
//             <Pn,Pn> = 2/(2n+1) if n >= 0.                                  //
//     This routine calculates Pn(x) using the following recursion:           //
//        (k+1) P[k+1](x) = (2k+1)x P[k](x) - k P[k-1](x), k = 1,...,n-1      //
//              P[0](x) = 1, P[1](x) = x.                                     //
//                                                                            //
//  Arguments:                                                                //
//     double x                                                               //
//        The argument of the Legendre polynomial Pn.                         //
//     int    n                                                               //
//        The degree of the Legendre polynomial Pn.                           //
//                                                                            //
//  Return Value:                                                             //
//     Pn(x) if n is a nonnegative integer.  If n is negative, 0 is returned. //
//                                                                            //
//  Example:                                                                  //
//     double Pn;                                                             //
//     double x;                                                              //
//     int    n;                                                              //
//                                                                            //
//     (user code to set x and n)                                             //
//                                                                            //
//     Pn = xLegendre_Pn(x, n);                                               //
////////////////////////////////////////////////////////////////////////////////

/**
* Berenger (contact at berenger dot eu)
* This is the source code to construct the legendre polynome in C
* This is fast but you can improve the code by using pointer instead of
* accessing using index on the array and to compute (2*l-1) with a recurrence.
* Ref: Fast and accurate determination of the Wigner rotation matrices in FMM
* url: http://berenger.eu/blog/c-legendre-polynomial-by-recurrence-programming/
*/

/**
* Compute the first term of the legendre polynome for the given value x and the polynomial degree n
*/
double getFirstLegendreTerm(const double x, const int n){
    if (n < 0){
        return 0.0;
    }

    if(fabs(x) == 1.0){
        if(x > 0.0 || n % 2 == 0){
            return 1.0;
        }
        return -1.0;
    }

    if (n == 0){
        return 1.0;
    }
    if (n == 1){
        return x;
    }

    double P0 = 1.0;
    double P1 = x;
    double Pn;

    for(int k = 1; k < n; k++){
        Pn = ((2 * k + 1) * x * P1 - (k * P0)) / (k + 1);
        P0 = P1;
        P1 = Pn;
    }

    return Pn;
}

#endif // FIRST_LEGENDRE_TERM_CL
