from mot.library_functions import SimpleCLLibrary

__author__ = 'Robbert Harms'
__date__ = '2018-09-12'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class FirstLegendreTerm(SimpleCLLibrary):
    def __init__(self):
        """Compute the first term of the legendre polynomial for the given value x and the polynomial degree n.

        The Legendre polynomials, Pn(x), are orthogonal on the interval [-1,1] with weight function w(x) = 1
        for -1 <= x <= 1 and 0 elsewhere.  They are normalized so that Pn(1) = 1.  The inner products are:

        .. code-block:: c

            <Pn,Pm> = 0        if n != m,
            <Pn,Pn> = 2/(2n+1) if n >= 0.


        This routine calculates Pn(x) using the following recursion:

        .. code-block:: c

            (k+1) P[k+1](x) = (2k+1)x P[k](x) - k P[k-1](x), k = 1,...,n-1
            P[0](x) = 1, P[1](x) = x.


        The function arguments are:

        * x: The argument of the Legendre polynomial Pn.
        * n: The degree of the Legendre polynomial Pn.

        The return value is Pn(x) if n is a non negative integer.  If n is negative, 0 is returned.
        """
        super().__init__('''
            double firstLegendreTerm(double x, int n){
                if (n < 0){
                    return 0.0;
                }

                if(fabs(x) == 1.0){
                    if(x > 0.0 || n % 2 == 0){
                        return 1.0;
                    }
                    return -1.0;
                }

                if (n == 0){
                    return 1.0;
                }
                if (n == 1){
                    return x;
                }

                double P0 = 1.0;
                double P1 = x;
                double Pn;

                for(uint k = 1; k < n; k++){
                    Pn = ((2 * k + 1) * x * P1 - (k * P0)) / (k + 1);
                    P0 = P1;
                    P1 = Pn;
                }

                return Pn;
            }
        ''')


class LegendreTerms(SimpleCLLibrary):
    def __init__(self):
        """Compute a range of Legendre terms for the given value x and the polynomial degree n.

        The Legendre polynomials, Pn(x), are orthogonal on the interval [-1,1] with weight function w(x) = 1
        for -1 <= x <= 1 and 0 elsewhere.  They are normalized so that Pn(1) = 1.  The inner products are:

        .. code-block:: c

            <Pn,Pm> = 0        if n != m,
            <Pn,Pn> = 2/(2n+1) if n >= 0.


        This routine calculates Pn(x) for all n in [0, 1, 2, ..., n] using the recursion:

        .. code-block:: c

            (k+1) P[k+1](x) = (2k+1)x P[k](x) - k P[k-1](x), k = 1,...,n-1
            P[0](x) = 1, P[1](x) = x.

        That is, this function will fill the given array legendre_terms with the values:
            [0] = firstLegendreTerm(x, 0)
            [1] = firstLegendreTerm(x, 1)
            [2] = firstLegendreTerm(x, 2)
            [3] = firstLegendreTerm(x, 3)
            ...

        The function arguments are:

            x: The argument of the Legendre polynomial Pn.
            n: The number of terms to fill.
            legendre_terms: an matrix of length n for storing the legendre terms
        """
        super().__init__('''
            void LegendreTerms(double x, uint n, double* legendre_terms){
                if(n <= 0){
                    return;
                }

                if(fabs(x) == 1.0){
                    for(uint i = 0; i < n; i++){
                        if(i % 2 == 0 || x > 0.0){
                            legendre_terms[i] = 1.0;    
                        }
                        else{
                            legendre_terms[i] = 1.0;
                        }
                    }
                }

                legendre_terms[0] = 1.0;

                double P0 = 1.0;
                double P1 = x;
                double Pn = P1;

                for(uint k = 1; k < n; k++){
                    legendre_terms[k] = Pn;

                    Pn = ((2 * k + 1) * x * P1 - (k * P0)) / (k + 1);
                    P0 = P1;
                    P1 = Pn;
                }
            }
        ''')


class EvenLegendreTerms(SimpleCLLibrary):
    def __init__(self):
        """Compute a range of even legendre terms for the given value x and the polynomial degree n.

        The Legendre polynomials, Pn(x), are orthogonal on the interval [-1,1] with weight function w(x) = 1
        for -1 <= x <= 1 and 0 elsewhere.  They are normalized so that Pn(1) = 1.  The inner products are:

        .. code-block:: c

            <Pn,Pm> = 0        if n != m,
            <Pn,Pn> = 2/(2n+1) if n >= 0.


        This routine calculates Pn(x) for all n in [0, 2, 4, ..., n] using the recursion:

        .. code-block:: c

            (k+1) P[k+1](x) = (2k+1)x P[k](x) - k P[k-1](x), k = 1,...,n-1
            P[0](x) = 1, P[1](x) = x.

        That is, this function will fill the given array legendre_terms with the values:
            [0] = firstLegendreTerm(x, 0)
            [1] = firstLegendreTerm(x, 2)
            [2] = firstLegendreTerm(x, 4)
            [3] = firstLegendreTerm(x, 8)
            ...

        The function arguments are:

            x: The argument of the Legendre polynomial Pn.
            n: The number of terms to fill.
            legendre_terms: an matrix of length n/2 for storing the even legendre terms
        """
        super().__init__('''
            void EvenLegendreTerms(double x, uint n, double* legendre_terms){
                if(n <= 0){
                    return;
                }

                if(fabs(x) == 1.0){
                    for(uint i = 0; i < n; i++){
                        legendre_terms[i] = 1.0;
                    }
                    return;
                }

                legendre_terms[0] = 1.0;

                double P0 = 1.0;
                double P1 = x;
                double Pn = P1;

                for(uint k = 1; k < n; k++){
                    Pn = ((2 * (2*k-1) + 1) * x * P1 - ((2*k-1) * P0)) / ((2*k-1) + 1);
                    P0 = P1;
                    P1 = Pn;

                    legendre_terms[k] = Pn;

                    Pn = ((2 * ((2*k-1)+1) + 1) * x * P1 - (((2*k-1)+1) * P0)) / (((2*k-1)+1) + 1);
                    P0 = P1;
                    P1 = Pn;
                }
            }
        ''')


class OddLegendreTerms(SimpleCLLibrary):
    def __init__(self):
        """Compute a range of odd legendre terms for the given value x and the polynomial degree n.

        The Legendre polynomials, Pn(x), are orthogonal on the interval [-1,1] with weight function w(x) = 1
        for -1 <= x <= 1 and 0 elsewhere.  They are normalized so that Pn(1) = 1.  The inner products are:

        .. code-block:: c

            <Pn,Pm> = 0        if n != m,
            <Pn,Pn> = 2/(2n+1) if n >= 0.


        This routine calculates Pn(x) for all n in [1, 3, 5, ..., n] using the recursion:

        .. code-block:: c

            (k+1) P[k+1](x) = (2k+1)x P[k](x) - k P[k-1](x), k = 1,...,n-1
            P[0](x) = 1, P[1](x) = x.

        That is, this function will fill the given array legendre_terms with the values:
            [0] = firstLegendreTerm(x, 1)
            [1] = firstLegendreTerm(x, 3)
            [2] = firstLegendreTerm(x, 5)
            [3] = firstLegendreTerm(x, 7)
            ...

        The function arguments are:

            x: The argument of the Legendre polynomial Pn.
            n: The number of terms to fill.
            legendre_terms: an matrix of length n/2 for storing the odd legendre terms
        """
        super().__init__('''
            void OddLegendreTerms(double x, uint n, double* legendre_terms){
                if(n <= 0){
                    return;
                }

                if(fabs(x) == 1.0){
                    if(x > 0.0){
                        for(uint i = 0; i < n; i++){
                            legendre_terms[i] = 1.0;
                        }
                    }
                    else{
                        for(uint i = 0; i < n; i++){
                            legendre_terms[i] = -1.0;
                        }
                    }
                    return;
                }

                double P0 = 1.0;
                double P1 = x;
                double Pn = P1;

                for(uint k = 1; k < n; k++){
                    legendre_terms[k-1] = Pn;

                    Pn = ((2 * (2*k-1) + 1) * x * P1 - ((2*k-1) * P0)) / ((2*k-1) + 1);
                    P0 = P1;
                    P1 = Pn;

                    Pn = ((2 * ((2*k-1)+1) + 1) * x * P1 - (((2*k-1)+1) * P0)) / (((2*k-1)+1) + 1);
                    P0 = P1;
                    P1 = Pn;
                }
            }
        ''')
