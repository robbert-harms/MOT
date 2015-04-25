from .base import AbstractSmoother
from ...utils import get_cl_double_extension_definer

__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MedianSmoother(AbstractSmoother):

    def __init__(self, size, cl_environments=None, load_balancer=None):
        super(MedianSmoother, self).__init__(size, cl_environments=cl_environments, load_balancer=load_balancer)

    def _get_kernel_source(self, volume_shape, cl_environment):
        kernel_source = get_cl_double_extension_definer(cl_environment.platform)
        kernel_source += self._get_ks_sub2ind_func(volume_shape)
        kernel_source += '''
            __kernel void filter(
                global double* volume,
                global char* mask,
                global double* results
                ){

                    ''' + self._get_ks_dimension_inits(len(volume_shape)) + '''
                    int ind = ''' + self._get_ks_sub2ind_func_call(len(volume_shape)) + ''';

                    if(mask[ind] > 0){
                        double voxels[''' + repr(self._calculate_length(len(volume_shape))) + '''];
                        int n = 0;

                        ''' + self._get_ks_dimension_sizes(volume_shape) + '''
                        ''' + self._get_ks_init_loop(volume_shape) + '''

                        if(n > 0){
                            int i;

                            double guess;
                            double maxltguess;
                            double mingtguess;
                            double less;
                            double greater;
                            double equal;
                            double minv = voxels[0];
                            double maxv = voxels[0];

                            for(i = 1; i < n; i++){
                                if(voxels[i] < minv){
                                    minv = voxels[i];
                                }
                                if(voxels[i] > maxv){
                                    maxv = voxels[i];
                                }
                            }

                            while(1){
                                guess = (minv+maxv)/2.0;
                                less = 0;
                                greater = 0;
                                equal = 0;
                                maxltguess = minv;
                                mingtguess = maxv;

                                for(i=0; i<n; i++){
                                    if(voxels[i] < guess){
                                        less++;
                                        if(voxels[i] > maxltguess){
                                            maxltguess = voxels[i];
                                        }
                                    }
                                    else if (voxels[i] > guess) {
                                        greater++;
                                        if(voxels[i] < mingtguess){
                                            mingtguess = voxels[i];
                                        }
                                    }
                                    else{
                                        equal++;
                                    }
                                }
                                if(less <= (n+1)/2 && greater <= (n+1)/2){
                                    break;
                                }
                                else if(less > greater){
                                    maxv = maxltguess;
                                }
                                else{
                                    minv = mingtguess;
                                }
                            }

                            if(less >= (n+1)/2){
                                guess = maxltguess;
                            }
                            else if(less + equal >= (n+1)/2){}
                            else{
                                guess = mingtguess;
                            }

                            results[ind] = guess;
                        }
                        else{
                            results[ind] = 0;
                        }
                    }
            }
        '''
        return kernel_source

    def _get_ks_init_loop(self, volume_shape):
        s = ''
        for i in range(len(volume_shape)):
            if i > 0:
                s += "\t" * (5 + len(volume_shape))
            s += 'for(dim' + repr(i) + ' = dim' + repr(i) + '_start; dim' + repr(i) + \
                    ' < dim' + repr(i) + '_end; dim' + repr(i) + '++){' + "\n"

        s += "\t" * (6 + len(volume_shape)) + 'if(mask[' + self._get_ks_sub2ind_func_call(len(volume_shape)) + '] > 0){'\
             + "\n"
        s += "\t" * (7 + len(volume_shape)) + 'voxels[n] = volume[' + self._get_ks_sub2ind_func_call(len(volume_shape))\
             + '];' + "\n"
        s += "\t" * (7 + len(volume_shape)) + 'n++;' + "\n"
        s += "\t" * (6 + len(volume_shape)) + '}' + "\n"
        for i in range(len(volume_shape)):
            s += "\t" * (5 + len(volume_shape)) + '}' + "\n"
        return s