/**
 * Author = Robbert Harms
 * Date = 2010-12-17
 * License = LGPL
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

#define BRACKET_GOLD 1.618034 /* the default ratio by which successive intervals are magnified in Bracketing */
#define BRACKET_EPSILON 30*MOT_EPSILON

/**
 * Bracket the minimum of a function.

    Given a function and distinct initial points, search in the
    downhill direction (as defined by the initial points) and return
    new points xa, xb, xc that bracket the minimum of the function
    f(xa) > f(xb) < f(xc). It doesn't always mean that obtained
    solution will satisfy xa<=x<=xb

    Inputs:
        xa, xb: Bracketing interval. As default, set `xa` to 0.0, and `xb` to 1.0.
        grow_limit: the grow limit, i.e. maximum magnification allowed for a parabolic-fit step in Bracketing
            set by default to 110.0

    Outputs:
        xa, xb, xc: Bracket.
        fa, fb, fc: Objective function values in bracket.
 */
void bracket%(SPF_NAME)s(mot_float_type* xa, mot_float_type* xb, mot_float_type* xc,
             mot_float_type* fa, mot_float_type* fb, mot_float_type* fc,
             void* eval_data, float grow_limit){

    mot_float_type w, fw, wlim, tmp, tmp1, tmp2, denom;

    *fa = %(FUNCTION_NAME)s(*xa, eval_data);
    *fb = %(FUNCTION_NAME)s(*xb, eval_data);

    if(*fa < *fb){
        tmp = *xb;
        *xb = *xa;
        *xa = tmp;

        tmp = *fb;
        *fb = *fa;
        *fa = tmp;
    }

    *xc = *xb + BRACKET_GOLD * (*xb - *xa);
    *fc = %(FUNCTION_NAME)s(*xc, eval_data);

    while(*fc < *fb){
        tmp1 = (*xb - *xa) * (*fb - *fc);
        tmp2 = (*xb - *xc) * (*fb - *fa);
        denom = tmp2 - tmp1;
        if(fabs(denom) < BRACKET_EPSILON){
            denom = 2 * BRACKET_EPSILON;
        }
        else{
            denom *= 2;
        }
        w = (*xb) - ((*xb - *xc) * tmp2 - (*xb - *xa) * tmp1) / denom;
        wlim = (*xb) + grow_limit * (*xc - *xb);

        if((w - *xc) * (*xb - w) > 0.0){
            fw = %(FUNCTION_NAME)s(w, eval_data);

            if(fw < *fc){
                *xa = *xb;
                *xb = w;
                *fa = *fb;
                *fb = fw;
                break;
            }
            else if(fw > *fb){
                *xc = w;
                *fc = fw;
                break;
            }
            w = (*xc) + BRACKET_GOLD * (*xc - *xb);
            fw = %(FUNCTION_NAME)s(w, eval_data);
        }
        else if((w - wlim) * (wlim - *xc) >= 0.0){
            w = wlim;
            fw = %(FUNCTION_NAME)s(w, eval_data);
        }
        else if((w - wlim) * (*xc - w) > 0.0){
            fw = %(FUNCTION_NAME)s(w, eval_data);

            if(fw < *fc){
                *xb = *xc;
                *xc = w;
                w = *xc + BRACKET_GOLD * (*xc - *xb);
                *fb = *fc;
                *fc = fw;
                fw = %(FUNCTION_NAME)s(w, eval_data);
            }
        }
        else{
            w = (*xc) + BRACKET_GOLD * (*xc - *xb);
            fw = %(FUNCTION_NAME)s(w, eval_data);
        }
        *xa = *xb;
        *xb = *xc;
        *xc = w;
        *fa = *fb;
        *fb = *fc;
        *fc = fw;
    }
}

#undef BRACKET_GOLD
#undef BRACKET_EPSILON
