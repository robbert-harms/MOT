__author__ = 'Robbert Harms'
__date__ = '2020-05-10'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'


from mot.library_functions import SimpleCLLibrary


class bessi0(SimpleCLLibrary):
    def __init__(self):
        """Return the zeroth-order modified Bessel function of the first kind

        Evaluate modified Bessel function In(x) and n=0.

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessi0(double x){
                double y;

                if(fabs(x) < 3.75){
                    y = (x / 3.75) * (x / 3.75);

                    return 1.0 + y * (3.5156229
                                      + y * (3.0899424
                                             + y * (1.2067492
                                                    + y * (0.2659732
                                                           + y * (0.360768e-1
                                                                  + y * 0.45813e-2)))));
                }

                y = 3.75 / fabs(x);
                return (exp(fabs(x)) / sqrt(fabs(x)))
                        * (0.39894228
                           + y * (0.1328592e-1
                                  + y * (0.225319e-2
                                         + y * (-0.157565e-2
                                                + y * (0.916281e-2
                                                       + y * (-0.2057706e-1
                                                              + y * (0.2635537e-1
                                                                     + y * (-0.1647633e-1
                                                                            + y * 0.392377e-2))))))));
            }
        ''')


class bessi1(SimpleCLLibrary):
    def __init__(self):
        """Return the first-order modified Bessel function of the first kind

        Evaluate modified Bessel function In(x) and n=1.

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessi1(double x){
                double ax,ans;
                double y;

                if ((ax=fabs(x)) < 3.75) {
                    y=x/3.75,y=y*y;
                    ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
                                +y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
                } else {
                    y=3.75/ax;
                    ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1-y*0.420059e-2));
                    ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2+y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
                    ans *= (exp(ax)/sqrt(ax));
                }
                return x < 0.0 ? -ans : ans;
            }
        ''')


class bessi(SimpleCLLibrary):
    def __init__(self):
        """Return the nth-order modified Bessel function of the first kind

        Evaluate modified Bessel function In(x) for n >= 0

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessi(double x, int n){
                double ACC = 40.0;
                double BIGNO = 1.0e10;
                double BIGNI = 1.0e-10;

                int j;
                double bi,bim,bip,tox,ans;

                if (n < 0){
                    return NAN;
                }
                if (n == 0){
                    return( bessi0(x) );
                }
                if (n == 1){
                    return( bessi1(x) );
                }

                if (x == 0.0){
                    return 0.0;
                }
                else {
                    tox=2.0/fabs(x);
                    bip=ans=0.0;
                    bi=1.0;
                    for (j=2*(n+(int) sqrt(ACC*n));j>0;j--) {
                        bim=bip+j*tox*bi;
                        bip=bi;
                        bi=bim;
                        if (fabs(bi) > BIGNO) {
                            ans *= BIGNI;
                            bi *= BIGNI;
                            bip *= BIGNI;
                        }
                        if (j == n) ans=bip;
                    }
                    ans *= bessi0(x)/bi;
                    return  x < 0.0 && n%2 == 1 ? -ans : ans;
                }
            }
        ''', dependencies=[bessi0(), bessi1()])


class log_bessi0(SimpleCLLibrary):
    def __init__(self):
        """Return the log of the zeroth-order modified Bessel function of the first kind."""
        super().__init__('''
            double log_bessi0(double x){
                if(x < 700){
                  return log(bessi0(x));
                }
                return x - log(2.0 * M_PI * x)/2.0;
            }
        ''', dependencies=(bessi0(),))


class nonexp_bessi(SimpleCLLibrary):
    def __init__(self):
        """Compute the modified Bessel functions of the first kind of order n, I_n(x), multiplied by :math:`e^{-x}`.

        Code inspired by [1], chapter 6.

        References:
        [1] Symbolic and Numeric Computation Series H. T. Lau - A Numerical Library in C for
            Scientists and Engineers-CRC Press (1995)
        """
        super().__init__('''
            double nonexp_bessi(double x, int n){
                if (n < 0){
                    return NAN;
                }
                if (x == 0.0) {
                    if(n == 0){
                        return 1.0;
                    }
                    return 0.0;
                } else {
                    return bessi(x, n) / exp(fabs(x));
                }
            }
        ''', dependencies=[bessel_starting_point(), bessi()])


class bessiaplusn(SimpleCLLibrary):
    def __init__(self):
        """Compute the modified Bessel functions of the first kind of order a+n.

        That is, it computes: :math:`I_{a+n}(x), (n >= 0, 0<=a<1)`.

        Code inspired by [1], chapter 6.

        References:
        [1] Symbolic and Numeric Computation Series H. T. Lau - A Numerical Library in C for
            Scientists and Engineers-CRC Press (1995)
        """
        super().__init__('''
            double bessiaplusn(double x, double a_n){
                double _n;
                double a = modf(a_n, &_n);
                int n = (int)_n;

                if(x == 0.0){
                    if(n == 0){
                        return (a == 0.0) ? 1.0 : 0.0;
                    }
                    return 0.0;
                } else if (a == 0.0) {
                    return bessi(x, n);
                } else if (a == 0.5) {
                    double c = 0.797884560802865 * sqrt(fabs(x)) * exp(fabs(x));
                    return nonexp_spher_bessi(x, n) * c;
                } else {
                    return nonexp_bessiaplusn(x, a_n) * exp(fabs(x));
                }
            }
        ''', dependencies=[bessi(), nonexp_spher_bessi(), nonexp_bessiaplusn()])


class nonexp_bessiaplusn(SimpleCLLibrary):
    def __init__(self):
        """Generates an array of modified Bessel functions of the first kind of order a+n, (n >= 0, O<=a<1),
            multiplied by the factor :math:`e^{-x}`.

        Code inspired by [1], chapter 6.

        References:
        [1] Symbolic and Numeric Computation Series H. T. Lau - A Numerical Library in C for
            Scientists and Engineers-CRC Press (1995)
        """
        super().__init__('''
            double nonexp_bessiaplusn(double x, double a_n){
                double _n;
                double a = modf(a_n, &_n);
                int n = (int)_n;

                if (x == 0.0) {
                     if(n == 0){
                        return (a == 0.0) ? 1.0 : 0.0;
                    }
                    return 0.0;
                } else if (a == 0.0) {
                    return nonexp_bessi(x, n);
                } else if (a == 0.5) {
                    double c = 0.797884560802865 * sqrt(fabs(x));
                    return nonexp_spher_bessi(x, n) * c;
                } else {
                    double i_tmp;
                    double l_begin;
                    int m,nu;
                    double r,s,labda,l,a2,x2;
                    a2=a+a;
                    x2=2.0/x;
                    nu=bessel_starting_point(x,n,1);

                    l = 1.0;
                    r = 0.0;
                    s = 0.0;

                    for (m=1; m<=nu; m++){
                        l=l*(m+a2)/(m+1);
                    }
                    l_begin = l;

                    for (m=nu; m>=1; m--) {
                        r=1.0/(x2*(a+m)+r);
                        l=l*(m+1)/(m+a2);
                        labda=l*(m+a)*2.0;
                        s=r*(labda+s);
                    }

                    i_tmp = 1.0/(1.0+s)/tgamma(1.0+a)/pow(x2,a);
                    for(m = 1; m <= n; m++){
                        l = l_begin;
                        r = 0.0;
                        s = 0.0;
                        for (int j=nu; j>=m; j--) {
                            r=1.0/(x2*(a+j)+r);
                            l=l*(j+1)/(j+a2);
                            labda=l*(j+a)*2.0;
                            s=r*(labda+s);
                        }
                        i_tmp *= r;
                    }
                    return i_tmp;
                }
            }
        ''', dependencies=[bessel_starting_point(), nonexp_bessi(), nonexp_spher_bessi()])


class nonexp_spher_bessi(SimpleCLLibrary):
    def __init__(self):
        """Calculates the modified spherical Bessel functions multiplied by :math:`e^{-x}`.

        Code inspired by [1], chapter 6.

        References:
        [1] Symbolic and Numeric Computation Series H. T. Lau - A Numerical Library in C for
            Scientists and Engineers-CRC Press (1995)
        """
        super().__init__('''
            double nonexp_spher_bessi(double x, int n){
                if (x == 0.0) {
                    if(n == 0){
                        return 1;
                    }
                    return 0;
                } else {
                    double r;
                    double i_tmp;
                    int start = bessel_starting_point(x, n, 1);

                    i_tmp = ((x == 0.0) ? 1.0 : (((2*x) < 0.7) ? sinh(x)/(x*exp(x)) : (1.0-exp(-(2*x)))/(2*x)));
                    for(int m = 1; m <= n; m++){
                        r = 0.0;
                        for(int j = start; j >= m; j--){
                            r = 1.0 / ((j+j+1)/x + r);
                        }
                        i_tmp *= r;
                    }
                    return i_tmp;
                }
            }
        ''', dependencies=[bessel_starting_point()])


class bessel_starting_point(SimpleCLLibrary):
    def __init__(self):
        """Generate a starting value for the Miller algorithm for computing an array of Bessel functions

        This is an auxiliary procedure which computes a starting value of an algorithm used in
        several Bessel function procedures.

        Code inspired by [1], chapter 6.

        Args:
            x: the argument of the Bessel functions, x > 0;
            n: the number of Bessel functions to be computed, n >= 0
            t: the type of Bessel function in question,
                t=O corresponds to ordinary Bessel functions;
                t=1 corresponds to modified Bessel functions.

        Returns:
            start: a starting value for the Miller algorithm for computing an array of Bessel functions;

        References:
        [1] Symbolic and Numeric Computation Series H. T. Lau - A Numerical Library in C for
            Scientists and Engineers-CRC Press (1995)
        """
        super().__init__('''
            int bessel_starting_point(double x, int n, int t){
                int s;
                double p,q,r,y;

                s=2*t-1;
                p=36.0/x-t;
                r=n/x;
                if (r > 1.0 || t == 1) {
                    q=sqrt(r*r+s);
                    r=r*log(q+r)-q;
                } else
                    r=0.0;
                q=18.0/x+r;
                r = (p > q) ? p : q;
                p=sqrt(2.0*(t+r));
                p=x*((1.0+r)+p)/(1.0+p);
                y=0.0;
                q=y;
                do {
                    y=p;
                    p /= x;
                    q=sqrt(p*p+s);
                    p=x*(r+q)/log(p+q);
                    q=y;
                } while (p > q || p < q-1.0);
                return ((t == 1) ? floor(p+1.0) : -floor(-p/2.0)*2);
            }
        ''')





