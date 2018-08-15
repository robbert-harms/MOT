from pkg_resources import resource_filename
from mot.library_functions import SimpleCLLibraryFromFile, SimpleCLLibrary

__author__ = 'Robbert Harms'
__date__ = '2018-05-12'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class CerfImWOfX(SimpleCLLibraryFromFile):
    def __init__(self):
        """Calculate the cerf."""
        super().__init__(
            'void', 'cerf', [],
            resource_filename('mot', 'data/opencl/cerf/im_w_of_x.cl'))


class dawson(SimpleCLLibrary):
    def __init__(self):
        super().__init__('double dawson(double x){ return (sqrt(M_PI)/2.0) * im_w_of_x(x); }',
                         dependencies=[CerfImWOfX()])


class erfi(SimpleCLLibrary):
    def __init__(self):
        """Calculate the imaginary error function for a real argument (special case).

        Compute erfi(x) = -i erf(ix), the imaginary error function.
        """
        super().__init__(
            'double erfi(double x){ return x*x > 720 ? (x > 0 ? INFINITY : -INFINITY) : exp(x*x) * im_w_of_x(x); }',
            dependencies=[CerfImWOfX()])
