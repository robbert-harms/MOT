__author__ = 'Robbert Harms'
__date__ = '2020-05-10'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'


from mot.library_functions import SimpleCLLibrary


class bessj0(SimpleCLLibrary):
    def __init__(self):
        """Evaluate ordinary Bessel function of first kind and order 0 at input x

        Evaluate ordinary Bessel function J_n(x) for n=0

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessj0( double x ){
                double ax,z;
                double xx,y,ans,ans1,ans2;

                if ((ax=fabs(x)) < 8.0) {
                    y=x*x;
                    ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7+y*(-11214424.18
                            +y*(77392.33017+y*(-184.9052456)))));
                    ans2=57568490411.0+y*(1029532985.0+y*(9494680.718+y*(59272.64853+y*(267.8532712+y*1.0))));
                    ans=ans1/ans2;
                } else {
                    z=8.0/ax;
                    y=z*z;
                    xx=ax-0.785398164;
                    ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4+y*(-0.2073370639e-5+y*0.2093887211e-6)));
                    ans2 = -0.1562499995e-1+y*(0.1430488765e-3+y*(-0.6911147651e-5
                            +y*(0.7621095161e-6-y*0.934935152e-7)));
                    ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
                }
                return ans;
            }
        ''')


class bessj1(SimpleCLLibrary):
    def __init__(self):
        """Evaluate ordinary Bessel function of first kind and order 1 at input x

        Evaluate ordinary Bessel function J_n(x) for n=1

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessj1( double x ){
                double ax,z;
                double xx,y,ans,ans1,ans2;

                if ((ax=fabs(x)) < 8.0) {
                    y=x*x;
                    ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
                            +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
                    ans2=144725228442.0+y*(2300535178.0+y*(18583304.74+y*(99447.43394+y*(376.9991397+y*1.0))));
                    ans=ans1/ans2;
                } else {
                    z=8.0/ax;
                    y=z*z;
                    xx=ax-2.356194491;
                    ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4+y*(0.2457520174e-5+y*(-0.240337019e-6))));
                    ans2=0.04687499995+y*(-0.2002690873e-3+y*(0.8449199096e-5+y*(-0.88228987e-6+y*0.105787412e-6)));
                    ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
                    if (x < 0.0) ans = -ans;
                }
                return ans;
            }
        ''')


class bessj(SimpleCLLibrary):
    def __init__(self):
        """Return the nth-order ordinary Bessel function of the first kind

        Evaluate ordinary Bessel function J_n(x) for n >= 0

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessj(int n, double x){
                double ACC = 40.0;
                double BIGNO = 1.0e10;
                double BIGNI = 1.0e-10;

                int    j, jsum, m;
                double ax, bj, bjm, bjp, sum, tox, ans;

                if (n < 0){
                    return NAN;
                }

                ax = fabs(x);
                if (n == 0){
                    return( bessj0(ax) );
                }
                if (n == 1){
                    return( bessj1(ax) );
                }

                if (ax == 0.0){
                    return 0.0;
                }
                else if (ax > (double) n) {
                    tox=2.0/ax;
                    bjm=bessj0(ax);
                    bj=bessj1(ax);
                    for (j=1;j<n;j++) {
                        bjp=j*tox*bj-bjm;
                        bjm=bj;
                        bj=bjp;
                    }
                    ans=bj;
                } else {
                    tox=2.0/ax;
                    m=2*((n+(int) sqrt(ACC*n))/2);
                    jsum=0;
                    bjp=ans=sum=0.0;
                    bj=1.0;
                    for (j=m;j>0;j--) {
                        bjm=j*tox*bj-bjp;
                        bjp=bj;
                        bj=bjm;
                        if (fabs(bj) > BIGNO) {
                            bj *= BIGNI;
                            bjp *= BIGNI;
                            ans *= BIGNI;
                            sum *= BIGNI;
                        }
                        if (jsum) sum += bj;
                        jsum=!jsum;
                        if (j == n) ans=bjp;
                    }
                    sum=2.0*sum-bj;
                    ans /= sum;
                }
                return  x < 0.0 && n%2 == 1 ? -ans : ans;
            }
        ''', dependencies=[bessj0(), bessj1()])


class bessy0(SimpleCLLibrary):

    def __init__(self):
        """Evaluate ordinary Bessel function of second kind and order 0 at input x

        Evaluate ordinary Bessel function Y_n(x) for n=0

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessy0( double x ){
                double z;
                double xx,y,ans,ans1,ans2;

                if (x < 8.0) {
                    y=x*x;
                    ans1 = -2957821389.0+y*(7062834065.0+y*(-512359803.6
                    +y*(10879881.29+y*(-86327.92757+y*228.4622733))));
                    ans2=40076544269.0+y*(745249964.8+y*(7189466.438
                    +y*(47447.26470+y*(226.1030244+y*1.0))));
                    ans=(ans1/ans2)+0.636619772*bessj0(x)*log(x);
                } else {
                    z=8.0/x;
                    y=z*z;
                    xx=x-0.785398164;
                    ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
                    +y*(-0.2073370639e-5+y*0.2093887211e-6)));
                    ans2 = -0.1562499995e-1+y*(0.1430488765e-3
                    +y*(-0.6911147651e-5+y*(0.7621095161e-6
                    +y*(-0.934945152e-7))));
                    ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
                }
                return ans;
            }
        ''', dependencies=[bessj0()])


class bessy1(SimpleCLLibrary):

    def __init__(self):
        """Evaluate ordinary Bessel function of second kind and order 1 at input x

        Evaluate ordinary Bessel function Y_n(x) for n=1

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessy1( double x ){
                double z;
                double xx,y,ans,ans1,ans2;

                if (x < 8.0) {
                    y=x*x;
                    ans1=x*(-0.4900604943e13+y*(0.1275274390e13+y*(-0.5153438139e11+y*(0.7349264551e9
                            +y*(-0.4237922726e7+y*0.8511937935e4)))));
                    ans2=0.2499580570e14+y*(0.4244419664e12+y*(0.3733650367e10+y*(0.2245904002e8
                            +y*(0.1020426050e6+y*(0.3549632885e3+y)))));
                    ans=(ans1/ans2)+0.636619772*(bessj1(x)*log(x)-1.0/x);
                } else {
                    z=8.0/x;
                    y=z*z;
                    xx=x-2.356194491;
                    ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4+y*(0.2457520174e-5+y*(-0.240337019e-6))));
                    ans2=0.04687499995+y*(-0.2002690873e-3+y*(0.8449199096e-5+y*(-0.88228987e-6+y*0.105787412e-6)));
                    ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
                }
                return ans;
            }
        ''', dependencies=[bessj1()])


class bessy(SimpleCLLibrary):
    def __init__(self):
        """Return the nth-order ordinary Bessel function of the second kind

        Evaluate ordinary Bessel function Y_n(x) for n>=0

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessy(int n, double x){
                int j;
                double by,bym,byp,tox;


                if (n < 0 || x == 0.0){
                    return NAN;
                }
                if (n == 0){
                    return( bessy0(x) );
                }
                if (n == 1){
                    return( bessy1(x) );
                }

                tox=2.0/x;
                by=bessy1(x);
                bym=bessy0(x);
                for (j=1;j<n;j++) {
                    byp=j*tox*by-bym;
                    bym=by;
                    by=byp;
                }
                return by;
            }
        ''', dependencies=[bessy0(), bessy1()])


class bessi0(SimpleCLLibrary):
    def __init__(self):
        """Return the zeroth-order modified Bessel function of the first kind

        Evaluate modified Bessel function I_n(x) and n=0.

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessi0(double x){
                double y;

                if(fabs(x) < 3.75){
                    y = (x / 3.75) * (x / 3.75);

                    return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1
                                        + y * 0.45813e-2)))));
                }
                y = 3.75 / fabs(x);
                return (exp(fabs(x)) / sqrt(fabs(x)))
                        * (0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
                                + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
                                    + y * 0.392377e-2))))))));
            }
        ''')


class bessi1(SimpleCLLibrary):
    def __init__(self):
        """Return the first-order modified Bessel function of the first kind

        Evaluate modified Bessel function I_n(x) and n=1.

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

        Evaluate modified Bessel function I_n(x) for n >= 0

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessi(int n, double x){
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


class bessk0(SimpleCLLibrary):
    def __init__(self):
        """Return the zeroth-order modified Bessel function of the second kind

        Evaluate modified Bessel function K_n(x) and n=0.

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessk0(double x){
                double y,ans;

                if (x <= 2.0) {
                    y=x*x/4.0;
                    ans=(-log(x/2.0)*bessi0(x))+(-0.57721566+y*(0.42278420+y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
                            +y*(0.10750e-3+y*0.74e-5))))));
                } else {
                    y=2.0/x;
                    ans=(exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1+y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
                                +y*(-0.251540e-2+y*0.53208e-3))))));
                }
                return ans;
            }
        ''', dependencies=[bessi0()])


class bessk1(SimpleCLLibrary):
    def __init__(self):
        """Return the first modified Bessel function of the second kind

        Evaluate modified Bessel function K_n(x) and n=1.

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessk1(double x){
                double y,ans;

                if (x <= 2.0) {
                    y=x*x/4.0;
                    ans=(log(x/2.0)*bessi1(x))+(1.0/x)*(1.0+y*(0.15443144+y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
                            +y*(-0.110404e-2+y*(-0.4686e-4)))))));
                } else {
                    y=2.0/x;
                    ans=(exp(-x)/sqrt(x))*(1.25331414+y*(0.23498619+y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
                            +y*(0.325614e-2+y*(-0.68245e-3)))))));
                }
                return ans;
            }
        ''', dependencies=[bessi1()])


class bessk(SimpleCLLibrary):
    def __init__(self):
        """Return the nth-order modified Bessel function of the second kind

        Evaluate modified Bessel function K_n(x) for n >= 0

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessk(int n, double x){
                int j;
                double bk,bkm,bkp,tox;

                if (n < 0 || x == 0.0){
                    return NAN;
                }
                if (n == 0){
                    return( bessk0(x) );
                }
                if (n == 1){
                    return( bessk1(x) );
                }

                tox=2.0/x;
                bkm=bessk0(x);
                bk=bessk1(x);
                for (j=1;j<n;j++) {
                    bkp=bkm+j*tox*bk;
                    bkm=bk;
                    bk=bkp;
                }
                return bk;
            }
        ''', dependencies=[bessk0(), bessk1()])


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


class nonexp_bessi0(SimpleCLLibrary):
    def __init__(self):
        """Return the zeroth-order modified Bessel function of the first kind multiplied by :math:`e^{-|x|}`

        Code inspired by [1], chapter 6.

        References:
        [1] Symbolic and Numeric Computation Series H. T. Lau - A Numerical Library in C for
            Scientists and Engineers-CRC Press (1995)
        """
        super().__init__('''
            double nonexp_bessi0(double x){
                if (x == 0.0) return 1.0;
                if (fabs(x) <= 15.0) {
                    return exp(-fabs(x))*bessi0(x);
                } else {
                    int i;
                    double sqrtx,br,br1,br2,z,z2,numerator,denominator;
                    double ar1[4]={0.2439260769778, -0.115591978104435e3, 0.784034249005088e4, -0.143464631313583e6};
                    double ar2[4]={1.0, -0.325197333369824e3, 0.203128436100794e5, -0.361847779219653e6};
                    x=fabs(x);
                    sqrtx=sqrt(x);
                    br1=br2=0.0;
                    z=30.0/x-1.0;
                    z2=z+z;
                    for (i=0; i<=3; i++) {
                        br=z2*br1-br2+ar1[i];
                        br2=br1;
                        br1=br;
                    }
                    numerator=z*br1-br2+0.346519833357379e6;
                    br1=br2=0.0;
                    for (i=0; i<=3; i++) {
                        br=z2*br1-br2+ar2[i];
                        br2=br1;
                        br1=br;
                    }
                    denominator=z*br1-br2+0.865665274832055e6;
                    return (numerator/denominator)/sqrtx;
                }
            }
        ''', dependencies=[bessi0()])


class nonexp_bessi1(SimpleCLLibrary):
    def __init__(self):
        """Return the first order modified Bessel function of the first kind multiplied by :math:`e^{-|x|}`

        Code inspired by [1], chapter 6.

        References:
        [1] Symbolic and Numeric Computation Series H. T. Lau - A Numerical Library in C for
            Scientists and Engineers-CRC Press (1995)
        """
        super().__init__('''
            double nonexp_bessi1(double x){
                if (x == 0.0) return 0.0;
                if (fabs(x) > 15.0) {
                    int i,signx;
                    double br,br1,br2,z,z2,sqrtx,numerator,denominator;
                    double ar1[4]={0.1494052814740e1, -0.362026420242263e3, 0.220549722260336e5, -0.408928084944275e6};
                    double ar2[4]={1.0, -0.631003200551590e3, 0.496811949533398e5, -0.100425428133695e7};
                    signx = (x > 0.0) ? 1 : -1;
                    x=fabs(x);
                    sqrtx=sqrt(x);
                    z=30.0/x-1.0;
                    z2=z+z;
                    br1=br2=0.0;
                    for (i=0; i<=3; i++) {
                        br=z2*br1-br2+ar1[i];
                        br2=br1;
                        br1=br;
                    }
                    numerator=z*br1-br2+0.102776692371524e7;
                    br1=br2=0.0;
                    for (i=0; i<=3; i++) {
                        br=z2*br1-br2+ar2[i];
                        br2=br1;
                        br1=br;
                    }
                    denominator=z*br1-br2+0.26028876789105e7;
                    return ((numerator/denominator)/sqrtx)*signx;
                } else {
                    return exp(-fabs(x))*bessi1(x);
                }
            }
        ''', dependencies=[bessi1()])


class nonexp_bessi(SimpleCLLibrary):
    def __init__(self):
        """Compute the modified Bessel functions of the first kind of order n, I_n(x), multiplied by :math:`e^{-x}`.

        Code inspired by [1], chapter 6.

        References:
        [1] Symbolic and Numeric Computation Series H. T. Lau - A Numerical Library in C for
            Scientists and Engineers-CRC Press (1995)
        """
        super().__init__('''
            double nonexp_bessi(int n, double x){
                if (n < 0){
                    return NAN;
                }
                if (x == 0.0) {
                    if(n == 0){
                        return 1.0;
                    }
                    return 0.0;
                } else {
                    return bessi(n, x) / exp(fabs(x));
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
            double bessiaplusn(double a_n, double x){
                double _n;
                double a = modf(a_n, &_n);
                int n = (int)_n;

                if(x == 0.0){
                    if(n == 0){
                        return (a == 0.0) ? 1.0 : 0.0;
                    }
                    return 0.0;
                } else if (a == 0.0) {
                    return bessi(n, x);
                } else if (a == 0.5) {
                    double c = 0.797884560802865 * sqrt(fabs(x)) * exp(fabs(x));
                    return nonexp_spher_bessi(n, x) * c;
                } else {
                    return nonexp_bessiaplusn(a_n, x) * exp(fabs(x));
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
            double nonexp_bessiaplusn(double a_n, double x){
                double _n;
                double a = modf(a_n, &_n);
                int n = (int)_n;

                if (x == 0.0) {
                     if(n == 0){
                        return (a == 0.0) ? 1.0 : 0.0;
                    }
                    return 0.0;
                } else if (a == 0.0) {
                    return nonexp_bessi(n, x);
                } else if (a == 0.5) {
                    double c = 0.797884560802865 * sqrt(fabs(x));
                    return nonexp_spher_bessi(n, x) * c;
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
            double nonexp_spher_bessi(int n, double x){
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

