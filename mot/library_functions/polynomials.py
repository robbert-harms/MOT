from mot.library_functions.base import SimpleCLLibrary

__author__ = 'Robbert Harms'
__date__ = '2018-05-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class real_zeros_cubic_pol(SimpleCLLibrary):

    def __init__(self):
        """Returns (only) the real roots of a cubic polynomial.

        This computes :math:`p(x) = \\sum_i c[i] * x^i = 0`, i.e. tries to find x such that :math:`p(x) = 0`
        using the algebraic method.

        The coefficients and the roots may point to the same address space to save memory.

        This code is an OpenCL translation from the Python code to be found at:
        https://github.com/shril/CubicEquationSolver/blob/master/CubicEquationSolver.py

        Args:
            coefficients: array of length 4, with the coefficients (a, b, c, d)
            roots: array of length 3, for the return values. Please note that only the first *n* values will be set,
                with n the number of returned real roots.

        Returns:
            the number of real roots
        """
        super().__init__('''
            int real_zeros_cubic_pol(double* coefficients, double* roots){
                const double a = coefficients[0];
                const double b = coefficients[1];
                const double c = coefficients[2];
                const double d = coefficients[3];
                
                if(a == 0 && b == 0){
                    roots[0] = -d / c;
                    return 1;
                }
                
                if(a == 0){
                    double D = c * c - 4.0 * b * d;
                    if(D >= 0){
                        D = sqrt(D);
                        roots[0] = (-c + D) / (2.0 * b);
                        roots[1] = (-c - D) / (2.0 * b);
                        return 2;
                    }
                    /* // Imaginary roots, not returned 
                        D = sqrt(-D);
                        roots[0] = (-c + D * 1j) / (2.0 * b);
                        roots[1] = (-c - D * 1j) / (2.0 * b);
                    */
                    return 0;
                }
                
                double f = ((3.0 * c / a) - ((b *b) / (a * a))) / 3.0;
                double g = (((2.0 * (b * b * b)) / (a * a * a)) - ((9.0 * b * c) / (a * a)) + (27.0 * d / a)) /27.0;
                double h = ((g * g) / 4.0 + (f * f * f) / 27.0);
                
                if(f == 0 && g == 0 && h == 0){
                    if((d / a) >= 0){
                        roots[0] = roots[1] = roots[2] = -pow(d/a, (double)(1/3.0));
                    }
                    else{
                        roots[0] = roots[1] = roots[2] = pow(-d/a, (double)(1/3.0));
                    }
                    return 3;
                }
                
                if(h <= 0){
                    const double i = sqrt(((g * g) / 4.0) - h);
                    const double j = pow(i, (double)(1/3.0));
                    const double k = acos(-(g / (2 * i)));
                    const double L = -j;
                    const double M = cos(k / 3.0);
                    const double N = sqrt(3.0) * sin(k / 3.0);
                    const double P = -(b / (3.0 * a));
            
                    roots[0] = 2 * j * cos(k / 3.0) - (b / (3.0 * a));
                    roots[1] = L * (M + N) + P;
                    roots[2] = L * (M - N) + P;
                    
                    return 3;
                }
                
                double R = -(g / 2.0) + sqrt(h);
                double S;
                if(R >= 0){
                    S = pow(R, (double)(1 / 3.0));
                }
                else{
                    S = -pow((-R), (double)(1 / 3.0));
                }
                
                double T = -(g / 2.0) - sqrt(h);
                double U;
                if(T >= 0){
                    U = pow(T, (double)(1 / 3.0));
                }
                else{
                    U = -pow((-T), (double)(1 / 3.0));
                }
                
                roots[0] = (S + U) - (b / (3.0 * a));
                /* //Imaginary roots, not returned
                roots[1] = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * sqrt(3) * 0.5j;
                roots[2] = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * sqrt(3) * 0.5j;
                */
                return 1;
            }
        ''')


class polevl(SimpleCLLibrary):

    def __init__(self):
        """Routines for computing polynomials.

        Copied from Scipy (https://github.com/scipy/scipy/blob/master/scipy/special/cephes/polevl.h), 05-05-2018.

        Evaluates polynomial of degree N::

                                2          N
            y  =  C  + C x + C x  +...+ C x
                   0      1     2          N

        Coefficients are stored in reverse order::

            coef[0] = C  , ..., coef[N] = C  .
                       N                   0
        """
        super().__init__('''
            double polevl(double x, double* coef, int N){
                double ans;
                double *p;
                int i;

                p = coef;
                ans = *p++;
                i = N;

                do
                    ans = ans * x  +  *p++;
                while( --i );

                return ans;
            }
        ''')


class p1evl(SimpleCLLibrary):

    def __init__(self):
        """Routines for computing polynomials when coefficient of x^N is 1.0.

        Copied from Scipy (https://github.com/scipy/scipy/blob/master/scipy/special/cephes/polevl.h), 05-05-2018.

        Evaluates polynomial of degree N::

                                2          N
            y  =  C  + C x + C x  +...+ C x
                   0      1     2          N

        Coefficients are stored in reverse order::

            coef[0] = C  , ..., coef[N] = C  .
                       N                   0

        In contrast to ``polevl``, this function assumes that coef[N] = 1.0 and is omitted from the array.
        Its calling arguments are otherwise the same as polevl().
        """
        super().__init__('''
            double p1evl(double x, double* coef, int N){
                double ans;
                double *p;
                int i;

                p = coef;
                ans = x + *p++;
                i = N-1;

                do
                    ans = ans * x  + *p++;
                while( --i );

                return( ans );
            }
        ''')


class ratevl(SimpleCLLibrary):

    def __init__(self):
        """Evaluates a rational integral.

        Copied from Scipy (https://github.com/scipy/scipy/blob/master/scipy/special/cephes/polevl.h), 2018-05-07.
        """
        super().__init__('''
            double ratevl(double x, double* num, int M, double* denom, int N){
                int i, dir;
                double y, num_ans, denom_ans;
                double absx = fabs(x);
                double *p;

                if (absx > 1) {
                    /* Evaluate as a polynomial in 1/x. */
                    dir = -1;
                    p = num + M;
                    y = 1 / x;
                } else {
                    dir = 1;
                    p = num;
                    y = x;
                }

                /* Evaluate the numerator */
                num_ans = *p;
                p += dir;
                for (i = 1; i <= M; i++) {
                    num_ans = num_ans * y + *p;
                    p += dir;
                }

                /* Evaluate the denominator */
                if (absx > 1) {
                    p = denom + N;
                } else {
                    p = denom;
                }

                denom_ans = *p;
                p += dir;
                for (i = 1; i <= N; i++) {
                    denom_ans = denom_ans * y + *p;
                    p += dir;
                }

                if (absx > 1) {
                    i = N - M;
                    return pow(x, i) * num_ans / denom_ans;
                } else {
                    return num_ans / denom_ans;
                }
            }
        ''')

