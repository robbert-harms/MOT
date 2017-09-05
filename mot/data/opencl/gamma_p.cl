/**
 * Author = Robbert Harms
 * Date = 2017-07-24
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/** Small number constant used in continued fraction gamma evaluation */
#define GAMMA_P_FPMIN 1E-30

/** Small number constant used in gamma series evaluation */
#define GAMMA_P_EPS 3E-7

/** Max number of iterations in series evaluation */
#define GAMMA_P_ITMAX 100

/**
 * Returns the incomplete gamma function P(a; x) evaluated by its series representation.
 */
double _gamma_p_using_series(const double a, const double x){
    double sum;
    double del;
    double ap;

    if(x <= 0.0){
        if (x < 0.0){
            return NAN;
        }
        return 0.0;
    }
    else{
        ap=a;
        del = sum = 1.0 / a;

        for(int n = 1; n <= GAMMA_P_ITMAX; n++){
            ++ap;
            del *= x/ap;
            sum += del;

            if(fabs(del) < fabs(sum) * GAMMA_P_EPS){
                return sum*exp(-x + a * log(x) - lgamma(a));
            }
        }
    }
    return NAN;
}

/*
 * Returns the incomplete gamma function Q(a; x) evaluated by its continued fraction representation.
 */
double _gamma_p_using_fraction(const double a, const double x){
    int i;
    double an,b,c,d,del,h;

    //Set up for evaluating continued fraction by modified Lentz's method (x5.2) with b0 = 0.
    b=x+1.0-a;
    c=1.0/GAMMA_P_FPMIN;
    d=1.0/b;
    h=d;
    for(i=1; i<=GAMMA_P_ITMAX; i++){
        an = -i*(i-a);
        b += 2.0;
        d=an*d+b;

        if(fabs(d) < GAMMA_P_FPMIN){
            d=GAMMA_P_FPMIN;
        }

        c=b+an/c;

        if(fabs(c) < GAMMA_P_FPMIN){
            c=GAMMA_P_FPMIN;
        }

        d=1.0/d;
        del=d*c;
        h *= del;

        if(fabs(del-1.0) < GAMMA_P_EPS){
            break;
        }
    }
    if(i > GAMMA_P_ITMAX){
        return NAN;
    }

    return exp(-x+a*log(x)-lgamma(a))*h;
}

/**
 * Calculates the normalized/regularized lower incomplete gamma function returning values in the range [0, 1].
 *
 * Both arguments must be positive.
 */
double gamma_p(const double a, const double x){
    if(x < 0.0 || a <= 0.0){
        return NAN;
    }

    if(x < (a + 1.0)){
        return _gamma_p_using_series(a, x);
    }

    return 1.0 - _gamma_p_using_fraction(a, x);
}

#undef GAMMA_P_FPMIN
#undef GAMMA_P_EPS
#undef GAMMA_P_ITMAX
