#ifndef CERF_DAWSON_CL
#define CERF_DAWSON_CL

/**
 * Author = robbert
 * Date = 2014-05-17
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

// sqrt(pi)/2
#define M_SQRTPI_2 0.8862269254527580

 /**
 * Calculate the Dawson's integral for a real argument.
 */
double dawson(double x){
    return M_SQRTPI_2 * im_w_of_x(x);
}

#undef M_SQRTPI_2
#endif // CERF_DAWSON_CL
