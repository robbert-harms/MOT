import pyopencl as cl
import numpy as np
from ...utils import get_float_type_def, split_in_batches
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker


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

        mle_values = np.zeros(samples.shape[0], dtype=np_dtype, order='C')
        map_values = np.zeros(samples.shape[0], dtype=np_dtype, order='C')

        workers = self._create_workers(
            lambda cl_environment: _MaxFinderWorker(
                cl_environment, self.get_compile_flags_list(model.double_precision), model,
                mle_indices, map_indices, mle_values, map_values))

        def process(samples_subset):
            for worker in workers:
                worker.set_samples(samples_subset)
            self.load_balancer.process(workers, samples.shape[0])

        # todo, set the current sample index

        max_batch_size = np.min([samples.shape[2], 1000])
        for batch_ind, batch_size in enumerate(split_in_batches(samples.shape[2], max_batch_size)):
            samples_subset = np.require(samples[..., (batch_ind * batch_size):((batch_ind + 1) * batch_size)],
                                        requirements=['C', 'A', 'O'])
            process(samples_subset)

        return mle_indices, map_indices, mle_values, map_values


class _MaxFinderWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model,
                 mle_indices, map_indices, mle_values, map_values):
        super(_MaxFinderWorker, self).__init__(cl_environment)

        self._model = model
        self._data_info = self._model.get_kernel_data()
        self._double_precision = model.double_precision

        self._samples = None
        self._state_arrays = [mle_indices, map_indices, mle_values, map_values]

        self._all_buffers = self._create_buffers()
        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def set_samples(self, samples):
        """Set the samples we will use for finding the maximum.

        One can repetitively set samples and recompute the workers to find the global maximum. This will
        read and write to and from the current maximum MLE and MAP values/indices found to find the global maximum.

        Args:
            samples (ndarray): the (d, p, n) matrix with the samples to search in this round
        """
        self._samples = samples

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        buffers = list(self._all_buffers)
        buffers.append(cl.Buffer(self._cl_run_context.context,
                                 cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                 hostbuf=self._samples))

        self._kernel.run_kernel(self._cl_run_context.queue, (int(nmr_problems),), None, *self._all_buffers,
                                global_offset=(int(range_start),))

        for buffer, state_array in zip(self._all_buffers, self._state_arrays):
            self._enqueue_readout(buffer, state_array, range_start, range_end)

    def _create_buffers(self):
        buffers = []
        for state_array in self._state_arrays:
            buffers.append(cl.Buffer(self._cl_run_context.context,
                                     cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                     hostbuf=state_array))
        return buffers

    def _get_kernel_source(self):
        ll_func = self._model.get_log_likelihood_per_observation_function()
        prior_func = self._model.get_log_prior_function(address_space_parameter_vector='private')

        cl_func = ''
        cl_func += ll_func.get_cl_code()
        cl_func += prior_func.get_cl_code()
        nmr_params = self._model.get_nmr_estimable_parameters()

        kernel_param_names = ['global uint* mle_ind',
                              'global uint* map_ind',
                              'global mot_float_type* mle_values',
                              'global mot_float_type* map_values',
                              'global mot_float_type* samples',
                              'global uint nmr_samples_per_problem',
                              ]

        kernel_param_names.extend(self._data_info.get_kernel_parameters())
        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._data_info.get_kernel_data_struct()
        kernel_source += cl_func

        kernel_source += '''
            double _calculate_log_likelihood(void* data, const mot_float_type* const x){
                double ll = 0;
                for(uint i = 0; i < ''' + str(self._model.get_nmr_inst_per_problem()) + '''; i++){
                    ll += ''' + ll_func.get_cl_function_name() + '''(data, x, i);
                }
                return ll;
            }
        '''

        kernel_source += '''
            __kernel void run_kernel(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ulong problem_ind = get_global_id(0);
                    
                    ''' + self._data_info.get_kernel_data_struct_initialization(
                            'data', problem_id_name='problem_ind') + '''
                    
                    double prior;
                    double ll; 
                    mot_float_type x[''' + str(nmr_params) + '''];
                    
                    for(uint sample_ind = 0; sample_ind < nmr_samples_per_problem; sample_ind++){
                        for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                            x[i] = samples[problem_ind * ''' + str(nmr_params) + ''' * nmr_samples_per_problem
                                           + i * nmr_samples_per_problem + sample_ind];
                        }

                        prior = ''' + prior_func.get_cl_function_name() + '''((void*)&data, x);
                        ll = _calculate_log_likelihood((void*)&data, x);
                    }
            }
        '''
        return kernel_source

