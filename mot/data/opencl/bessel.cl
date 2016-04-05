/**
 * Author = Robbert Harms
 * Date = 2016-02-09
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/*

    Copied from: bessel.c

                      Copyright (c) 1998
                  Kapteyn Institute Groningen
                     All Rights Reserved.
*/

/*
#>            bessel.dc2

Function:     BESSEL
Purpose:      Evaluate Bessel function J, Y, I, K of integer order.
Category:     MATH
File:         bessel.c
Author:       M.G.R. Vogelaar
Use:          See bessj.dc2, bessy.dc2, bessi.dc2 or bessk.dc2
Description:  The differential equation

                       2
                   2  d w       dw      2   2
                  x . --- + x . --- + (x - v ).w = 0
                        2       dx
                      dx

              has two solutions called Bessel functions of the first kind
              Jv(x) and Bessel functions of the second kind Yv(x).
              The routines bessj and bessy return the J and Y for
              integer v and therefore are called Bessel functions
              of integer order.

              The differential equation

                       2
                   2  d w       dw      2   2
                  x . --- + x . --- - (x + v ).w = 0
                        2       dx
                      dx

              has two solutions called modified Bessel functions
              Iv(x) and Kv(x).
              The routines bessi and bessk return the I and K for
              integer v and therefore are called Modified Bessel
              functions of integer order.
              (Abramowitz & Stegun, Handbook of mathematical
              functions, ch. 9, pages 358,- and 374,- )

              The implementation is based on the ideas from
              Numerical Recipes, Press et. al.
              This routine is NOT callable in FORTRAN.

Updates:      Jun 29, 1998: VOG, Document created.
#<
*/

/*------------------------------------------------------------*/
/* Zeroth-order modified Bessel function of the first kind.   */
/*------------------------------------------------------------*/
double bessel_i0(double x){
    double y;

    if(fabs(x) < 3.75f){
      y = (x/3.75) * (x/3.75);
      return 1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
    }

    y=3.75/fabs(x);
    return (exp(fabs(x))/sqrt(fabs(x)))*(0.39894228+y*(0.1328592e-1
         +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
         +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
         +y*0.392377e-2))))))));
}

/**
 * Return the log of the zeroth-order modified Bessel function of the first kind.
 */
double log_bessel_i0(double x){
    if(x < 700){
        return log(bessel_i0(x));
    }
    return x - log(2.0 * M_PI * x)/2.0;
}
