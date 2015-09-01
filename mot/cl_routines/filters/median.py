from .base import AbstractFilter, AbstractFilterWorker
from ...utils import get_cl_pragma_double, get_float_type_def

__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MedianFilter(AbstractFilter):

    def _get_worker(self, *args):
        """Create the worker that we will use in the computations.

        This is supposed to be overwritten by the implementing filterer.

        Returns:
            the worker object
        """
        return _MedianFilterWorker(self, *args)


class _MedianFilterWorker(AbstractFilterWorker):

    def _get_kernel_source(self):
        kernel_source = get_cl_pragma_double()
        kernel_source += get_float_type_def(self._use_double, 'masking_float')
        kernel_source += self._get_ks_sub2ind_func(self._volume_shape)
        kernel_source += '''
            __kernel void filter(
                global masking_float* volume,
                ''' + ('global char* mask,' if self._use_mask else '') + '''
                global masking_float* results
                ){

                    ''' + self._get_ks_dimension_inits(len(self._volume_shape)) + '''
                    const int ind = ''' + self._get_ks_sub2ind_func_call(len(self._volume_shape)) + ''';

                    ''' + ('if(mask[ind] > 0){' if self._use_mask else 'if(true){') + '''

                        ''' + self._get_ks_dimension_sizes(self._volume_shape) + '''

                        masking_float guess;
                        masking_float maxltguess;
                        masking_float mingtguess;
                        masking_float less;
                        masking_float greater;
                        masking_float equal;
                        masking_float minv = volume[ind];
                        masking_float maxv = volume[ind];
                        int number_of_items = 0;

                        masking_float tmp_val = 0.0;

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
                                    less += 1;
                                    if(tmp_val > maxltguess){
                                        maxltguess = tmp_val;
                                    }
                                }
                                else if (tmp_val > guess) {
                                    greater += 1;
                                    if(tmp_val < mingtguess){
                                        mingtguess = tmp_val;
                                    }
                                }
                                else{
                                    equal += 1;
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
            s += 'for(dim' + str(i) + ' = dim' + str(i) + '_start; dim' + str(i) + \
                    ' < dim' + str(i) + '_end; dim' + str(i) + '++){' + "\n"

        if self._use_mask:
            s += 'if(mask[' + self._get_ks_sub2ind_func_call(len(self._volume_shape)) + '] > 0){' + "\n"

        s += body

        if self._use_mask:
            s += '}' + "\n"

        for i in range(len(self._volume_shape)):
            s += '}' + "\n"
        return s