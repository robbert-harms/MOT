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
#ifndef M_SQRTPI_2
#define M_SQRTPI_2 0.8862269254527580
#endif

 /**
 * Calculate the Dawson's integral for a real argument.
 */
double dawson(double x){
    return M_SQRTPI_2 * im_w_of_x(x);
}

float fdawson(float x){
    return M_SQRTPI_2 * fim_w_of_x(x);
}

#endif // CERF_DAWSON_CL
