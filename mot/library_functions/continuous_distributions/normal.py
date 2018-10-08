from mot.library_functions import polevl, p1evl
from mot.library_functions.base import SimpleCLLibrary

__author__ = 'Robbert Harms'
__date__ = '2018-05-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class normal_pdf(SimpleCLLibrary):

    def __init__(self):
        """Compute the Probability Density Function of the Gaussian distribution."""
        super().__init__('''
            double normal_pdf(double x, double mean, double std){
                return exp(-((x - mean) * (x - mean)) / (2 * std * std)) / sqrt(2 * M_PI * std * std);
            }
        ''')


class normal_logpdf(SimpleCLLibrary):

    def __init__(self):
        """Compute the log of the Probability Density Function of the Gaussian distribution."""
        super().__init__('''
            double normal_logpdf(double x, double mean, double std){
                return -((x - mean) * (x - mean)) / (2 * std * std) - (log(std) + (0.5 * log(2 * M_PI)));
            }
        ''')


class normal_cdf(SimpleCLLibrary):

    def __init__(self):
        """Compute the Cumulative Distribution Function of the Gaussian distribution."""
        super().__init__('''
            double normal_cdf(double x, double mean, double std){
                return (1 + erf((x - mean) / (std * M_SQRT2))) / 2.0;
            }
        ''')


class normal_ppf(SimpleCLLibrary):

    def __init__(self):
        """Computes the inverse of the cumulative distribution function of the Gaussian distribution.

        This is the inverse of the Gaussian CDF.
        """
        super().__init__('''
            double normal_ppf(double y, double mean, double std){
                return _ndtri(y) * std + mean;
            }''', dependencies=(_ndtri(),))


class _ndtri(SimpleCLLibrary):
    def __init__(self):
        """Inverse of Normal distribution function.

        Code taken from Scipy (https://github.com/scipy/scipy/blob/master/scipy/special/cephes/NDTRI.c), 05-05-2018.

        Returns the argument, x, for which the area under the Gaussian probability density function (integrated from
        minus infinity to x) is equal to y.

        For small arguments 0 < y < exp(-2), the program computes z = sqrt( -2.0 * log(y) );  then the approximation is
        x = z - log(z)/z  - (1/z) P(1/z) / Q(1/z). There are two rational functions P/Q, one for 0 < y < exp(-32)
        and the other for y up to exp(-2).  For larger arguments,
        w = y - 0.5, and  x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2)).
        """
        super().__init__('''
            double _ndtri(double y){
                /* approximation for 0 <= |y - 0.5| <= 3/8 */
                double P0[5] = {
                    -5.99633501014107895267E1,
                    9.80010754185999661536E1,
                    -5.66762857469070293439E1,
                    1.39312609387279679503E1,
                    -1.23916583867381258016E0,
                };

                double Q0[8] = {
                    /*  1.00000000000000000000E0, */
                    1.95448858338141759834E0,
                    4.67627912898881538453E0,
                    8.63602421390890590575E1,
                    -2.25462687854119370527E2,
                    2.00260212380060660359E2,
                    -8.20372256168333339912E1,
                    1.59056225126211695515E1,
                    -1.18331621121330003142E0,
                };

                /* Approximation for interval z = sqrt(-2 log y ) between 2 and 8
                 * i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
                 */
                double P1[9] = {
                    4.05544892305962419923E0,
                    3.15251094599893866154E1,
                    5.71628192246421288162E1,
                    4.40805073893200834700E1,
                    1.46849561928858024014E1,
                    2.18663306850790267539E0,
                    -1.40256079171354495875E-1,
                    -3.50424626827848203418E-2,
                    -8.57456785154685413611E-4,
                };

                double Q1[8] = {
                    /*  1.00000000000000000000E0, */
                    1.57799883256466749731E1,
                    4.53907635128879210584E1,
                    4.13172038254672030440E1,
                    1.50425385692907503408E1,
                    2.50464946208309415979E0,
                    -1.42182922854787788574E-1,
                    -3.80806407691578277194E-2,
                    -9.33259480895457427372E-4,
                };

                /* Approximation for interval z = sqrt(-2 log y ) between 8 and 64
                 * i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
                 */

                double P2[9] = {
                    3.23774891776946035970E0,
                    6.91522889068984211695E0,
                    3.93881025292474443415E0,
                    1.33303460815807542389E0,
                    2.01485389549179081538E-1,
                    1.23716634817820021358E-2,
                    3.01581553508235416007E-4,
                    2.65806974686737550832E-6,
                    6.23974539184983293730E-9,
                };

                double Q2[8] = {
                    /*  1.00000000000000000000E0, */
                    6.02427039364742014255E0,
                    3.67983563856160859403E0,
                    1.37702099489081330271E0,
                    2.16236993594496635890E-1,
                    1.34204006088543189037E-2,
                    3.28014464682127739104E-4,
                    2.89247864745380683936E-6,
                    6.79019408009981274425E-9,
                };

                double x, y1, z, y2, x0, x1;
                int code;

                if (y <= 0.0){
                    return -INFINITY;
                }
                if (y >= 1.0){
                    return INFINITY;
                }

                code = 1;
                y1 = y;
                if (y1 > (1.0 - 0.13533528323661269189)) {	/* 0.135... = exp(-2) */
                    y1 = 1.0 - y1;
                    code = 0;
                }

                if (y1 > 0.13533528323661269189) {
                    y1 = y1 - 0.5;
                    y2 = y1 * y1;
                    x = y1 + y1 * (y2 * polevl(y2, P0, 4) / p1evl(y2, Q0, 8));
                    x = x * 2.50662827463100050242E0;  /* sqrt(2pi) */
                    return (x);
                }

                x = sqrt(-2.0 * log(y1));
                x0 = x - log(x) / x;

                z = 1.0 / x;
                if (x < 8.0) { /* y1 > exp(-32) = 1.2664165549e-14 */
                    x1 = z * polevl(z, P1, 8) / p1evl(z, Q1, 8);
                } 
                else{
                    x1 = z * polevl(z, P2, 8) / p1evl(z, Q2, 8);
                }

                x = x0 - x1;

                if (code != 0){
                    x = -x;
                }

                return (x);
            }
        ''', dependencies=(polevl(), p1evl()))

