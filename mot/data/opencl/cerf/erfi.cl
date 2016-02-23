#ifndef CERF_ERFI_CL
#define CERF_ERFI_CL

/**
 * Author = robbert
 * Date = 2014-05-17
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Calculate the imaginary error function for a real argument (special case)
 */
double erfi(double x){
    // Compute erfi(x) = -i erf(ix),
    // the imaginary error function.
    return x*x > 720 ? (x > 0 ? INFINITY : -INFINITY) : exp(x*x) * im_w_of_x(x);
}

float ferfi(float x){
    // Compute erfi(x) = -i erf(ix),
    // the imaginary error function.
    return x*x > 720 ? (x > 0 ? INFINITY : -INFINITY) : exp(x*x) * im_w_of_x(x);
}

#endif //CERF_ERFI_CL
