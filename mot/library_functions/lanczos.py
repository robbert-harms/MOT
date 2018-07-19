from mot.library_functions import SimpleCLLibrary, ratevl

__author__ = 'Robbert Harms'
__date__ = '2018-05-14'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class lanczos_sum_expg_scaled(SimpleCLLibrary):

    def __init__(self):
        """

        Copied from Scipy (https://github.com/scipy/scipy/blob/master/scipy/special/cephes/lanczos.c), 2018-05-07.
        """
        super().__init__('''
            double lanczos_sum_expg_scaled(double x){
                double lanczos_sum_expg_scaled_num[13] = {
                    0.006061842346248906525783753964555936883222,
                    0.5098416655656676188125178644804694509993,
                    19.51992788247617482847860966235652136208,
                    449.9445569063168119446858607650988409623,
                    6955.999602515376140356310115515198987526,
                    75999.29304014542649875303443598909137092,
                    601859.6171681098786670226533699352302507,
                    3481712.15498064590882071018964774556468,
                    14605578.08768506808414169982791359218571,
                    43338889.32467613834773723740590533316085,
                    86363131.28813859145546927288977868422342,
                    103794043.1163445451906271053616070238554,
                    56906521.91347156388090791033559122686859
                };
                
                double lanczos_sum_expg_scaled_denom[13] = {
                    1,
                    66,
                    1925,
                    32670,
                    357423,
                    2637558,
                    13339535,
                    45995730,
                    105258076,
                    150917976,
                    120543840,
                    39916800,
                    0
                };
                    
                return ratevl(x, lanczos_sum_expg_scaled_num, 13 - 1, lanczos_sum_expg_scaled_denom, 13 - 1);
            }
        ''', dependencies=(ratevl(),))
