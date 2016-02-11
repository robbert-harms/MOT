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
    return pown(x, 2) > 720 ? (x > 0 ? INFINITY : -INFINITY) : exp(pown(x, 2)) * im_w_of_x((double)x);
}

float ferfi(float x){
    // Compute erfi(x) = -i erf(ix),
    // the imaginary error function.
    return pown(x, 2) > 720 ? (x > 0 ? INFINITY : -INFINITY) : exp(pown(x, 2)) * fim_w_of_x((float)x);
}

#endif //CERF_ERFI_CL
