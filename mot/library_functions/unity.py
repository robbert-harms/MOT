from mot.library_functions.base import SimpleCLLibrary

__author__ = 'Robbert Harms'
__date__ = '2018-05-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class log1pmx(SimpleCLLibrary):
    def __init__(self):
        """log(1 + x) - x"""
        super().__init__('''
            double log1pmx(double x){
                if (fabs(x) < 0.5) {
                    int n;
                    double xfac = x;
                    double term;
                    double res = 0;

                    for(n = 2; n < 500; n++) {
                        xfac *= -x;
                        term = xfac / n;
                        res += term;
                        if (fabs(term) < MOT_EPSILON * fabs(res)) {
                            break;
                        }
                    }
                    return res;
                }
                else {
                    return log1p(x) - x;
                }
            }
        ''')


class lgam1p(SimpleCLLibrary):
    def __init__(self):
        """Compute lgam(x + 1).

        This is a simplification of the corresponding function in scipy
        https://github.com/scipy/scipy/blob/master/scipy/special/cephes/unity.c 2018-05-14
        """
        super().__init__('double lgam1p(double x){return lgamma(x + 1);}')
