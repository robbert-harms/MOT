from mot.library_functions.base import SimpleCLLibrary

__author__ = 'Robbert Harms'
__date__ = '2018-05-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class polevl(SimpleCLLibrary):

    def __init__(self):
        """Routines for computing polynomials.

        Copied from Scipy (https://github.com/scipy/scipy/blob/master/scipy/special/cephes/polevl.h), 05-05-2018.

        Evaluates polynomial of degree N:

                            2          N
        y  =  C  + C x + C x  +...+ C x
               0      1     2          N

        Coefficients are stored in reverse order:

        coef[0] = C  , ..., coef[N] = C  .
                   N                   0
        """
        super(polevl, self).__init__(
            'double', 'polevl', [('double', 'x'), ('double*', 'coef'), ('int', 'N')],
            '''
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
            ''')


class p1evl(SimpleCLLibrary):

    def __init__(self):
        """Routines for computing polynomials when coefficient of x^N is 1.0.

        Copied from Scipy (https://github.com/scipy/scipy/blob/master/scipy/special/cephes/polevl.h), 05-05-2018.

        Evaluates polynomial of degree N:

                            2          N
        y  =  C  + C x + C x  +...+ C x
               0      1     2          N

        Coefficients are stored in reverse order:

        coef[0] = C  , ..., coef[N] = C  .
                   N                   0

        In contrast to ``polevl``, this function assumes that coef[N] = 1.0 and is omitted from the array.
        Its calling arguments are otherwise the same as polevl().
        """
        super(p1evl, self).__init__(
            'double', 'p1evl', [('double', 'x'), ('double*', 'coef'), ('int', 'N')],
            '''
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
            ''')


class ratevl(SimpleCLLibrary):

    def __init__(self):
        """Evaluates a rational integral.

        Copied from Scipy (https://github.com/scipy/scipy/blob/master/scipy/special/cephes/polevl.h), 2018-05-07.
        """
        super(ratevl, self).__init__(
            'double', 'ratevl',
            [('double', 'x'),
             ('double*', 'num'),
             ('int', 'M'),
             ('double*', 'denom'),
             ('int', 'N')],
            '''
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
            ''')

