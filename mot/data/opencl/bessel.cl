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
/* Zeroth-order Bessel function of the first kind.            */
/*------------------------------------------------------------*/
double bessel_j0(double x){
   double ax = fabs(x);
   double z;
   double xx,y,ans,ans1,ans2;

   if(ax < 8.0){
      y=x*x;
      ans1 = 57568490574.0+y*(-13362590354.0+y*(651619640.7 +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
      ans2 = 57568490411.0+y*(1029532985.0+y*(9494680.718 +y*(59272.64853+y*(267.8532712+y*1.0))));
      ans=ans1/ans2;
   } else {
      z=8.0/ax;
      y=z*z;
      xx=ax-0.785398164;
      ans1 = 1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4 + y*(-0.2073370639e-5+y*0.2093887211e-6)));
      ans2 = -0.1562499995e-1+y*(0.1430488765e-3 +y*(-0.6911147651e-5+y*(0.7621095161e-6 -y*0.934935152e-7)));
      ans = sqrt(0.636619772 / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2);
   }

   return ans;
}


/*------------------------------------------------------------*/
/* Zeroth-order modified Bessel function of the first kind.   */
/*------------------------------------------------------------*/
double bessel_i0(double x){
    double ax = fabs(x);
    double y;

    if(ax < 3.75){
      y = pown(x/3.75, 2);
      return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y *
                                                                                    (0.360768e-1 + y * 0.45813e-2)))));
    }

    y=3.75/ax;
    return (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1
     + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
     + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
     + y * 0.392377e-2))))))));
}

/**
 * Return the log of the zeroth-order modified Bessel function of the first kind.
 */
double log_bessel_i0(double x){
    if(x < 700){
        return log((double)bessel_i0(x));
    }
    return x - log(2.0 * M_PI * x)/2.0;
}
