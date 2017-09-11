import numpy as np
from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import KernelInputBuffer, SimpleNamedCLFunction, KernelInputScalar
from ...cl_routines.base import CLRoutine


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Sampling_MLE_MAP_Index(CLRoutine):

    def calculate(self, model, samples):
        """Calculate the (two sets of) indices of the parameters that maximize the likelihood and the a posteriori.

        This calculates the log likelihoods and posterior value for every sample of every problem in the sampling chain
        and returns for every problem the with indices of the maximum log likelihoods and of the maximum a posteriori.
        These can subsequently be used to get point estimates for every parameter (by taking the parameters
        corresponding to the index of these maxima).

        Args:
            model (AbstractModel): The model to use for the likelihood and prior models
            samples (ndarray): the (d, p, n) matrix with d problems, p parameters and n samples.

        Returns:
            tuple: (ndarray, ndarray, ndarray, ndarray): maps for each problem instance, with first the maximum
                likelihood index, second the maximum a posteriori index, third the maximum likelihood value and
                fourth the maximum a posteriori value.
        """
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        samples = np.require(samples, np_dtype, requirements=['C', 'A', 'O'])

        mle_indices = np.zeros(samples.shape[0], dtype=np.uint32, order='C')
        map_indices = np.zeros(samples.shape[0], dtype=np.uint32, order='C')

        mle_values = np.ones(samples.shape[0], dtype=np_dtype, order='C') * -np.inf
        map_values = np.ones(samples.shape[0], dtype=np_dtype, order='C') * -np.inf

        all_kernel_data = dict(model.get_kernel_data())
        all_kernel_data.update({
            'samples': KernelInputBuffer(samples),
            'mle_indices': KernelInputBuffer(mle_indices, is_readable=False, is_writable=True),
            'map_indices': KernelInputBuffer(map_indices, is_readable=False, is_writable=True),
            'mle_values': KernelInputBuffer(mle_values, is_readable=False, is_writable=True),
            'map_values': KernelInputBuffer(map_values, is_readable=False, is_writable=True),
            'nmr_params': KernelInputScalar(samples.shape[1]),
            'nmr_samples': KernelInputScalar(samples.shape[2])
        })

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._get_wrapped_function(model), all_kernel_data, samples.shape[0],
                             double_precision=model.double_precision)

        return (all_kernel_data['mle_indices'].get_data(),
                all_kernel_data['map_indices'].get_data(),
                all_kernel_data['mle_values'].get_data(),
                all_kernel_data['map_values'].get_data())

    def _get_wrapped_function(self, model):
        ll_func = model.get_log_likelihood_per_observation_function()
        prior_func = model.get_log_prior_function(address_space_parameter_vector='private')
        nmr_params = model.get_nmr_estimable_parameters()

        func = ''
        func += ll_func.get_cl_code()
        func += prior_func.get_cl_code()

        func += '''
            double _calculate_log_likelihood(mot_data_struct* data, const mot_float_type* const x){
                double ll = 0;
                for(uint i = 0; i < ''' + str(model.get_nmr_inst_per_problem()) + '''; i++){
                    ll += ''' + ll_func.get_cl_function_name() + '''(data, x, i);
                }
                return ll;
            }
        '''
        func += '''
            void compute(mot_data_struct* data){
                double prior;
                double ll;
                mot_float_type x[''' + str(nmr_params) + '''];

                for(uint sample_ind = 0; sample_ind < data->nmr_samples; sample_ind++){
                    for(uint i = 0; i < data->nmr_params; i++){
                        x[i] = data->samples[i * data->nmr_samples + sample_ind];
                    }

                    prior = ''' + prior_func.get_cl_function_name() + '''(data, x);
                    ll = _calculate_log_likelihood(data, x);

                    if(ll > *(data->mle_values)){
                        *(data->mle_values) = ll;
                        *(data->mle_indices) = sample_ind;
                    }

                    if(ll + prior > *(data->map_values)){
                        *(data->map_values) = ll + prior;
                        *(data->map_indices) = sample_ind;
                    }
                }
            }
        '''
        return SimpleNamedCLFunction(func, 'compute')
