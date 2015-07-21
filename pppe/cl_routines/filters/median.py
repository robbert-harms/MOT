from .base import AbstractFilter, AbstractFilterWorker
from ...utils import get_cl_double_extension_definer

__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MedianFilter(AbstractFilter):

    def __init__(self, size, cl_environments=None, load_balancer=None):
        super(MedianFilter, self).__init__(size, cl_environments=cl_environments, load_balancer=load_balancer)

    def _get_worker(self, *args):
        """Create the worker that we will use in the computations.

        This is supposed to be overwritten by the implementing filterer.

        Returns:
            the worker object
        """
        return _MedianFilterWorker(self, *args)


class _MedianFilterWorker(AbstractFilterWorker):

    def _get_kernel_source(self):
        kernel_source = get_cl_double_extension_definer(self._cl_environment.platform)
        kernel_source += self._get_ks_sub2ind_func(self._volume_shape)
        kernel_source += '''
            __kernel void filter(
                global double* volume,
                ''' + ('global char* mask,' if self._use_mask else '') + '''
                global double* results
                ){

                    ''' + self._get_ks_dimension_inits(len(self._volume_shape)) + '''
                    const int ind = ''' + self._get_ks_sub2ind_func_call(len(self._volume_shape)) + ''';

                    ''' + ('if(mask[ind] > 0){' if self._use_mask else 'if(true){') + '''

                        ''' + self._get_ks_dimension_sizes(self._volume_shape) + '''

                        double guess;
                        double maxltguess;
                        double mingtguess;
                        double less;
                        double greater;
                        double equal;
                        double minv = volume[ind];
                        double maxv = volume[ind];
                        int number_of_items = 0;

                        double tmp_val = 0.0;

                        ''' + self._loop_encapsulate('''
                            tmp_val = volume[''' +
                                    self._get_ks_sub2ind_func_call(len(self._volume_shape)) + '''];

                            if(tmp_val < minv){
                                minv = tmp_val;
                            }
                            if(tmp_val > maxv){
                                maxv = tmp_val;
                            }

                            number_of_items++;
                        ''') + '''

                        while(1){
                            guess = (minv+maxv)/2.0;
                            less = 0;
                            greater = 0;
                            equal = 0;
                            maxltguess = minv;
                            mingtguess = maxv;

                            ''' + self._loop_encapsulate('''
                                tmp_val = volume[''' +
                                    self._get_ks_sub2ind_func_call(len(self._volume_shape)) + '''];
                                if(tmp_val < guess){
                                    less++;
                                    if(tmp_val > maxltguess){
                                        maxltguess = tmp_val;
                                    }
                                }
                                else if (tmp_val > guess) {
                                    greater++;
                                    if(tmp_val < mingtguess){
                                        mingtguess = tmp_val;
                                    }
                                }
                                else{
                                    equal++;
                                }
                            ''') + '''
                            if(less <= (number_of_items + 1)/2 && greater <= (number_of_items + 1)/2){
                                break;
                            }
                            else if(less > greater){
                                maxv = maxltguess;
                            }
                            else{
                                minv = mingtguess;
                            }
                        }

                        if(less >= (number_of_items + 1)/2){
                            guess = maxltguess;
                        }
                        else if(less + equal >= (number_of_items + 1)/2){}
                        else{
                            guess = mingtguess;
                        }

                        results[ind] = guess;
                    }
            }
        '''
        return kernel_source

    def _loop_encapsulate(self, body):
        s = ''
        for i in range(len(self._volume_shape)):
            s += 'for(dim' + repr(i) + ' = dim' + repr(i) + '_start; dim' + repr(i) + \
                    ' < dim' + repr(i) + '_end; dim' + repr(i) + '++){' + "\n"

        if self._use_mask:
            s += 'if(mask[' + self._get_ks_sub2ind_func_call(len(self._volume_shape)) + '] > 0){' + "\n"

        s += body

        if self._use_mask:
            s += '}' + "\n"

        for i in range(len(self._volume_shape)):
            s += '}' + "\n"
        return s