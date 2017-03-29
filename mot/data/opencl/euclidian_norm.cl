#ifndef EUCLIDIAN_NORM_%(MEMSPACE)s_%(MEMTYPE)s_CL
#define EUCLIDIAN_NORM_%(MEMSPACE)s_%(MEMTYPE)s_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-01
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

#ifndef ENORM_SQRT_GIANT
#define ENORM_SQRT_GIANT sqrt(DBL_MAX) /* square should not overflow */
#endif

#ifndef ENORM_SQRT_DWARF
#define ENORM_SQRT_DWARF sqrt(DBL_MIN) /* square should not underflow */
#endif

/*****************************************************************************/
/*  euclidian_norm (Euclidean norm)                                          */
/*****************************************************************************/
%(MEMTYPE)s euclidian_norm_%(MEMSPACE)s(const %(MEMSPACE)s %(MEMTYPE)s* const x, const int n){
/*     Given an n-vector x, this function calculates the
 *     euclidean norm of x.
 *
 *     The euclidean norm is computed by accumulating the sum of
 *     squares in three different sums. The sums of squares for the
 *     small and large components are scaled so that no overflows
 *     occur. Non-destructive underflows are permitted. Underflows
 *     and overflows do not occur in the computation of the unscaled
 *     sum of squares for the intermediate components.
 *     The definitions of small, intermediate and large components
 *     depend on two constants, LM_SQRT_DWARF and LM_SQRT_GIANT. The main
 *     restrictions on these constants are that LM_SQRT_DWARF**2 not
 *     underflow and LM_SQRT_GIANT**2 not overflow.
 *
 *     Parameters
 *
 *      n is a positive integer input variable.
 *
 *      x is an input array of length n.
 */
    int i;
    %(MEMTYPE)s s1, s2, s3, xabs, x1max, x3max, sqrt_n_tmp;

    s1 = 0;
    s2 = 0;
    s3 = 0;
    x1max = 0;
    x3max = 0;
    sqrt_n_tmp = ENORM_SQRT_GIANT / n;

    /** sum squares. **/
    for (i = 0; i < n; i++) {
        xabs = fabs(x[i]);

        if (xabs > ENORM_SQRT_DWARF) {
            if ( xabs < sqrt_n_tmp ) {
                s2 += xabs * xabs;
            }
            else if ( xabs > x1max ) {
                s1 = s1 * ((x1max / xabs) * (x1max / xabs)) + 1;
                x1max = xabs;
            }
            else {
                s1 += ((xabs / x1max) * (xabs / x1max));
            }
        }
        else if ( xabs > x3max ) {
            s3 = s3 * ((x3max / xabs) * (x3max / xabs)) + 1;
            x3max = xabs;
        }
        else if (xabs != 0.) {
            s3 += ((xabs / x3max) * (xabs / x3max));
        }
    }

    /** calculation of norm. **/
    if (s1 != 0){
        return x1max * sqrt(s1 + (s2 / x1max) / x1max);
    }
    else if(s2 != 0){
        if(s2 >= x3max){
            return sqrt(s2 * (1 + (x3max / s2) * (x3max * s3)));
        }
        else{
            return sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
        }
    }
    else{
        return x3max * sqrt(s3);
    }

} /*** euclidian_norm. ***/

#endif // EUCLIDIAN_NORM_%(MEMSPACE)s_%(MEMTYPE)s_CL
