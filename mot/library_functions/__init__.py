import os

from mot.lib.cl_function import SimpleCLCodeObject
from mot.library_functions.base import SimpleCLLibrary, SimpleCLLibraryFromFile, CLLibrary
from pkg_resources import resource_filename

from mot.library_functions.eispack import eispack_tred2, eispack_tql2
from mot.library_functions.unity import log1pmx
from mot.library_functions.polynomials import p1evl, polevl, ratevl, real_zeros_cubic_pol
from mot.library_functions.continuous_distributions.normal import normal_cdf, normal_pdf, normal_logpdf, normal_ppf
from mot.library_functions.continuous_distributions.gamma import gamma_pdf, gamma_logpdf, gamma_ppf, gamma_cdf, \
    gamma_cdf_approx, gamma_ppf_approx
from mot.library_functions.continuous_distributions.invgamma import invgamma_pdf, invgamma_logpdf, \
    invgamma_cdf, invgamma_ppf
from mot.library_functions.error_functions import dawson, CerfImWOfX, erfi
from mot.library_functions.legendre_polynomial import FirstLegendreTerm, LegendreTerms, \
    EvenLegendreTerms, OddLegendreTerms
from mot.library_functions.special_functions import bessi0, bessi1, log_bessi0, bessi, bessiaplusn, \
    nonexp_bessi, nonexp_bessiaplusn, nonexp_spher_bessi, bessel_starting_point


__author__ = 'Robbert Harms'
__date__ = '2018-05-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'


class LogCosh(SimpleCLLibrary):
    def __init__(self):
        """Computes :math:`log(cosh(x))`

        For large x this will try to estimate it without overflow. For small x we use the opencl functions log and cos.
        The estimation for large numbers has been taken from:
        https://github.com/JaneliaSciComp/tmt/blob/master/basics/logcosh.m
        """
        super().__init__('''
            double log_cosh(double x){
                if(x < 50){
                    return log(cosh(x));
                }
                return fabs(x) + log(1 + exp(-2.0 * fabs(x))) - log(2.0);
            }
        ''')


class Rand123(SimpleCLCodeObject):
    def __init__(self):
        generator = 'philox'

        src = open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/openclfeatures.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/array.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/{}.h'.format(generator)), ),
                    'r').read()
        src += (open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/rand123.h'), ), 'r').read() % {
            'GENERATOR_NAME': (generator)
        })
        super().__init__(src)


class EuclidianNormFunction(SimpleCLLibraryFromFile):
    def __init__(self, memspace='private', memtype='mot_float_type'):
        """A CL functions for calculating the Euclidian distance between n values.

        Args:
            memspace (str): The memory space of the memtyped array (private, constant, global).
            memtype (str): the memory type to use, double, float, mot_float_type, ...
        """
        super().__init__(
            memtype,
            self.__class__.__name__ + '_' + memspace + '_' + memtype,
            [],
            resource_filename('mot', 'data/opencl/euclidian_norm.cl'),
            var_replace_dict={'MEMSPACE': memspace, 'MEMTYPE': memtype})


class simpsons_rule(SimpleCLLibrary):

    def __init__(self, function_name):
        """Create a CL function for integrating a function using Simpson's rule.

        This creates a CL function specifically meant for integrating the function of the given name.
        The name of the generated CL function will be 'simpsons_rule_<function_name>'.

        Args:
            function_name (str): the name of the function to integrate, accepting the arguments:
                - a: the lower bound of the integral
                - b: the upper bound of the integral
                - n: the number of steps, i.e. the number of approximations to make
                - data: a pointer to some data, this is passed on to the function we are integrating.
        """
        super().__init__('''
            double simpsons_rule_{f}(double a, double b, uint n, void* data){{
                double h = (b - a) / n;

                double sum_odds = {f}(a + h/2.0, data);
                double sum_evens = 0.0;

                double x;

                for(uint i = 1; i < n; i++){{
                    sum_odds += {f}(a + h * i + h / 2.0, data);
                    sum_evens += {f}(a + h * i, data);
                }}

                return h / 6.0 * ({f}(a, data) + {f}(b, data) + 4.0 * sum_odds + 2.0 * sum_evens);
            }}
        '''.format(f=function_name))


class linear_cubic_interpolation(SimpleCLLibrary):

    def __init__(self):
        """Cubic interpolation for a one-dimensional grid.

        This uses the theory of Cubic Hermite splines for interpolating a one-dimensional grid of values.

        At the borders, it will clip the values to the nearest border.

        For more information on this method, see https://en.wikipedia.org/wiki/Cubic_Hermite_spline.

        Example usage:
            constant float data[] = {1.0, 2.0, 5.0, 6.0};
            linear_cubic_interpolation(1.5, 4, data);
        """
        super().__init__('''
            double linear_cubic_interpolation(double x, int y_len, constant float* y_values){
                int n = x;
                double u = x - n;

                double p0 = y_values[min(max((int)0, n - 1), y_len - 1)];
                double p1 = y_values[min(max((int)0, n    ), y_len - 1)];
                double p2 = y_values[min(max((int)0, n + 1), y_len - 1)];
                double p3 = y_values[min(max((int)0, n + 2), y_len - 1)];

                double a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
                double b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
                double c = 0.5 * (-p0 + p2);
                double d = p1;

                return d + u * (c + u * (b + u * a));
            }
        ''')


class eigenvalues_3x3_symmetric(SimpleCLLibrary):

    def __init__(self):
        """Calculate the eigenvalues of a symmetric 3x3 matrix.

        This simple algorithm only works in case of a real and symmetric matrix.

        The input to this function is an array with the upper triangular elements of the matrix.
        The output are the eigenvalues such that eig1 >= eig2 >= eig3, i.e. from large to small.

        Args:
            A: the upper triangle of an 3x3 matrix
            v: the output eigenvalues as a vector of three elements.

        References:
            [1]: https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
            [2]: Smith, Oliver K. (April 1961), "Eigenvalues of a symmetric 3 × 3 matrix.", Communications of the ACM,
                 4 (4): 168, doi:10.1145/355578.366316
        """
        super().__init__('''
            void eigenvalues_3x3_symmetric(double* A, double* v){
                double p1 = (A[1] * A[1]) + (A[2] * A[2]) + (A[4] * A[4]);

                if (p1 == 0.0){
                    v[0] = A[0];
                    v[1] = A[3];
                    v[2] = A[5];
                    return;
                }

                double q = (A[0] + A[3] + A[5]) / 3.0;
                double p = sqrt(((A[0] - q)*(A[0] - q) + (A[3] - q)*(A[3] - q) + (A[5] - q)*(A[5] - q) + 2*p1) / 6.0);

                double r = (
                     ((A[0] - q)/p) * ((((A[3] - q)/p) * ((A[5] - q)/p)) - ((A[4]/p) * (A[4]/p)))
                    - (A[1]/p)      * ((A[1]/p) * ((A[5] - q)/p) - (A[2]/p) * (A[4]/p))
                    + (A[2]/p)      * ((A[1]/p) * (A[4]/p) - (A[2]/p) * ((A[3] - q)/p))
                ) / 2.0;

                double phi;
                if(r <= -1){
                    phi = M_PI / 3.0;
                }
                else if(r >= 1){
                    phi = 0;
                }
                else{
                    phi = acos(r) / 3;
                }

                v[0] = q + 2 * p * cos(phi);
                v[2] = q + 2 * p * cos(phi + (2*M_PI/3.0));
                v[1] = 3 * q - v[0] - v[2];
            }
        ''')


class multiply_square_matrices(SimpleCLLibrary):
    def __init__(self):
        """Matrix multiplication of two square matrices.

        Having this as a special function is slightly faster than a more generic matrix multiplication algorithm.

        All matrices are expected to be in c/row-major order.

        Parameters:
            n: the rectangular size of the matrix (the number of rows / the number of columns).
            A[n*n]: the left matrix
            B[n*n]: the right matrix
            C[n*n]: the output matrix
        """
        super().__init__('''
            void multiply_square_matrices(uint n, double* A, double* B, double* C){
                int i, j, z;
                double sum;
                for(int ind = 0; ind < n*n; ind++){
                    i = ind / n;
                    j = ind - n * i;

                    sum = 0;
                    for(z = 0; z < n; z++){
                        sum += A[i * n + z] * B[z * n + j];
                    }
                    C[ind] = sum;
                }
            }
        ''')


class eigen_decompose_real_symmetric_matrix(SimpleCLLibrary):

    def __init__(self):
        """Computes eigenvalues and eigenvectors of real symmetric matrix.

        This uses the RS routine from the EISPACK code, to be found at:
        https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.html

        It first tri-diagonalizes the matrix using Householder transformations and then computes the diagonal
        using QR transformations.

        This routine only works with real symmetric matrices as input.

        Parameters:
            n: input, the order of the matrix.
            A[n*n]: input, the real symmetric matrix to invert
            W[n]: output, the eigenvalues in ascending order.
            Z[n*n]: output, the eigenvectors, can coincide with A[n*n].
            scratch[n]: input, scratch data

        Output:
            error code: the error for the the TQL2 routine (see EISPACK).
            The no-error, normal completion code is zero.
        """
        super().__init__('''
            int eigen_decompose_real_symmetric_matrix(int n, double* A, double* w, double* Z, double* scratch){

                // tri-diagonalize the matrix
                eispack_tred2 ( n, A, w, scratch, Z );

                // QR decompose the tri-diagonal
                return eispack_tql2 ( n, w, scratch, Z );
            }
        ''', dependencies=[eispack_tred2(), eispack_tql2()])


class pseudo_inverse_real_symmetric_matrix_upper_triangular(SimpleCLLibrary):

    def __init__(self):
        """Compute the pseudo-inverse of a real symmetric matrix stored as an upper triangular vector.

        Results are placed in the upper triangular input vector.

        Parameters:
            n: the size of the symmetric matrix
            A[n*(n+1)/2]: on input, the matrix to inverse. On output, the inverse of the matrix.
            scratch[2*n + 2*n*n]: scratch data
        """
        super().__init__('''
            void pseudo_inverse_real_symmetric_matrix_upper_triangular(uint n, double* A, double* scratch){
                int i, j, z, ind_counter;
                double sum;

                double* w = scratch;
                double* Z = w + n;
                double* decomposition_scratch = Z + n * n;
                double* Z_transpose_w = decomposition_scratch + n;

                ind_counter = 0;
                for (j = 0; j < n; j++){
                    for (i = j; i < n; i++){
                        Z[i+j*n] = A[ind_counter];
                        ind_counter++;
                    }
                }

                eigen_decompose_real_symmetric_matrix(n, Z, w, Z, decomposition_scratch);

                // inverse the eigenvalues
                for(i = 0; i < n; i++){
                    if(w[i] == 0.){
                        w[i] = 0;
                    }
                    else{
                        w[i] = fabs(1/w[i]);
                    }
                }

                // prepare the left matrix Z.T*w
                for(i = 0; i < n; i++){
                    for(j = 0; j < n; j++){
                        Z_transpose_w[i * n + j] = Z[j * n + i] * w[j];
                    }
                }

                // matrix multiplication of (Z.T*w) * Z
                ind_counter = 0;
                for(i = 0; i < n; i++){
                    for(j = i; j < n; j++){

                        sum = 0;
                        for(z = 0; z < n; z++){
                            sum += Z_transpose_w[i * n + z] * Z[z * n + j];
                        }

                        A[ind_counter] = sum;
                        ind_counter++;
                    }
                }
            }
        ''', dependencies=[eigen_decompose_real_symmetric_matrix()])
