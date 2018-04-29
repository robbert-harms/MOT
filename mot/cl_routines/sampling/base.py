import logging
from contextlib import contextmanager

from mot.cl_routines.base import CLRoutine
from mot.cl_routines.mapping.run_procedure import RunProcedure
from mot.library_functions import Rand123
from mot.utils import split_in_batches, get_float_type_def, NameFunctionTuple
from mot.kernel_data import KernelScalar, KernelLocalMemory, KernelArray, \
    KernelAllocatedArray
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2014-05-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractSampler(CLRoutine):

    def __init__(self, model, starting_positions, **kwargs):
        """Abstract base class for sampling routines.

        Sampling routines implementing this interface should be stateful objects that, for the given model, keep track
        of the sampling state over multiple calls to :meth:`sample`.

        Args:
            model (SampleModelInterface): the model to sample.
            starting_positions (ndarray): the starting positions for the sampler. Should be a two dimensional matrix
                with for every modeling instance (first dimension) and every parameter (second dimension) a value.
        """
        super(AbstractSampler, self).__init__(**kwargs)
        self._logger = logging.getLogger(__name__)
        self._model = model
        self._starting_positions = starting_positions
        if len(starting_positions.shape) < 2:
            self._starting_positions = self._starting_positions[..., None]
        self._nmr_problems = self._model.get_nmr_problems()
        self._nmr_params = self._starting_positions.shape[1]
        self._sampling_index = 0
        self._current_chain_position = np.require(np.copy(self._starting_positions),
                                                  requirements=['C', 'A', 'O', 'W'],
                                                  dtype=self._cl_runtime_info.mot_float_dtype)
        self._rng_state = np.random.uniform(low=np.iinfo(np.uint32).min, high=np.iinfo(np.uint32).max + 1,
                                            size=(self._nmr_problems, 6)).astype(np.uint32)

        if self._starting_positions.shape[0] != model.get_nmr_problems():
            raise ValueError('The number of problems in the model does not match the number of starting points given.')

        if self._starting_positions.shape[1] != model.get_nmr_parameters():
            raise ValueError('The number of parameters in the model does not match the number of '
                             'starting points given.')

    def sample(self, nmr_samples, burnin=0, thinning=1):
        """Take additional samples from the given model using this sampler.

        This method can be called multiple times in which the sampling state is stored in between.

        Args:
            nmr_samples (int): the number of samples to return
            burnin (int): the number of samples to discard before returning samples
            thinning (int): how many sample we wait before storing a new one. This will draw extra samples such that
                    the total number of samples generated is ``nmr_samples * (thinning)`` and the number of samples
                    stored is ``nmr_samples``. If set to one or lower we store every sample after the burn in.

        Returns:
            SamplingOutput: the sampling output object
        """
        if not thinning or thinning < 1:
            thinning = 1
        if not burnin or burnin < 0:
            burnin = 0

        with self._logging(nmr_samples, burnin, thinning):
            if burnin > 0:
                for batch_start, batch_end in split_in_batches(burnin, max(1000 // thinning, 100)):
                    self._sample(batch_end - batch_start, return_output=False)
            if nmr_samples > 0:
                outputs = []
                for batch_start, batch_end in split_in_batches(nmr_samples, max(1000 // thinning, 100)):
                    outputs.append(self._sample(batch_end - batch_start, thinning=thinning))
                return SimpleSampleOutput(*[np.concatenate([o[ind] for o in outputs], axis=-1) for ind in range(3)])

    def _sample(self, nmr_samples, thinning=1, return_output=True):
        """Sample the given number of samples with the given thinning.

        If ``return_output`` we will return the samples, log likelihoods and log priors. If not, we will advance the
        state of the sampler without returning storing the samples.

        Args:
            nmr_samples (int): the number of iterations to advance the sampler
            thinning (int): the thinning to apply
            return_output (boolean): if we should return the output

        Returns:
            None or tuple: if ``return_output`` is True three ndarrays as (samples, log_likelihoods, log_priors)
        """
        kernel_data = self._get_kernel_data(nmr_samples, thinning, return_output)

        runner = RunProcedure(self._cl_runtime_info)
        runner.run_procedure(self._get_compute_func(nmr_samples, thinning, return_output), kernel_data,
                             self._nmr_problems, use_local_reduction=True)
        self._sampling_index += nmr_samples * thinning

        self._readout_kernel_data(kernel_data)

        if return_output:
            return (kernel_data['_samples'].get_data(),
                    kernel_data['_log_likelihoods'].get_data(),
                    kernel_data['_log_priors'].get_data())

    def _get_kernel_data(self, nmr_samples, thinning, return_output):
        """Get the kernel data we will input to the MCMC sampler.

        By default, this will take the kernel data from the model and add to that the items:

        * _nmr_iterations: the number of iterations to sample
        * _iteration_offset: the current sampling index, that is, the offset to the given number of iterations
        * _rng_state: the random number generator state
        * _current_chain_position: the current position of the sampled chain
        * _log_likelihood_tmp: an OpenCL local array for combining the log likelihoods

        Additionally, if ``return_output`` is True, we add to that the arrays:

        * _samples: for the samples
        * _log_likelihoods: for storing the log likelihoods
        * _log_priors: for storing the priors

        Args:
            nmr_samples (int): the number of samples we will draw
            thinning (int): the thinning factor we want to use
            return_output (boolean): if the kernel should return output

        Returns:
            dict[str: mot.utils.KernelData]: the kernel input data
        """
        kernel_data = dict(self._model.get_kernel_data())
        kernel_data.update({
            '_nmr_iterations': KernelScalar(nmr_samples * thinning, ctype='ulong'),
            '_iteration_offset': KernelScalar(self._sampling_index, ctype='ulong'),
            '_rng_state': KernelArray(self._rng_state, 'uint', is_writable=True, ensure_zero_copy=True),
            '_current_chain_position': KernelArray(self._current_chain_position, 'mot_float_type',
                                                   is_writable=True, ensure_zero_copy=True),
            '_log_likelihood_tmp': KernelLocalMemory('double')
        })

        if return_output:
            kernel_data.update({
                '_samples': KernelAllocatedArray((self._nmr_problems, self._nmr_params, nmr_samples),
                                                       'mot_float_type'),
                '_log_likelihoods': KernelAllocatedArray((self._nmr_problems, nmr_samples), 'mot_float_type'),
                '_log_priors': KernelAllocatedArray((self._nmr_problems, nmr_samples), 'mot_float_type'),
            })
        return kernel_data

    def _readout_kernel_data(self, kernel_data):
        """Readout the kernel data and update the sampler state with the state from the compute device.

        Args:
            kernel_data (dict[str: mot.utils.KernelData]): the kernel data from which to read the output
        """
        pass

    def _get_compute_func(self, nmr_samples, thinning, return_output):
        """Get the MCMC algorithm as a computable function.

        Args:
            nmr_samples (int): the number of samples we will draw
            thinning (int): the thinning factor we want to use
            return_output (boolean): if the kernel should return output

        Returns:
            mot.utils.NameFunctionTuple: the compute function
        """
        kernel_source = '''
                    #define NMR_OBSERVATIONS ''' + str(self._model.get_nmr_observations()) + '''
                '''
        kernel_source += get_float_type_def(self._cl_runtime_info.double_precision)
        random_library = Rand123()
        kernel_source += random_library.get_cl_code()

        kernel_source += self._get_log_prior_cl_func()
        kernel_source += self._get_log_likelihood_cl_func()

        kernel_source += self._get_state_update_cl_func(nmr_samples, thinning, return_output)

        kernel_source += '''
            void compute(mot_data_struct* data){
                bool is_first_work_item = get_local_id(0) == 0;
    
                rand123_data rand123_rng_data = rand123_initialize_data((uint[]){
                    data->_rng_state[0], data->_rng_state[1], data->_rng_state[2], data->_rng_state[3], 
                    data->_rng_state[4], data->_rng_state[5], 0, 0});
                void* rng_data = (void*)&rand123_rng_data;
    
                local mot_float_type current_position[''' + str(self._nmr_params) + '''];
                local double current_likelihood;
                local mot_float_type current_prior;
    
                if(is_first_work_item){
                    for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        current_position[i] = data->_current_chain_position[i];
                    }
                    current_prior = _computeLogPrior(data, current_position);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
    
                _computeLogLikelihood(data, current_position, data->_log_likelihood_tmp, &current_likelihood);
    
                for(ulong i = 0; i < data->_nmr_iterations; i++){
        '''
        if return_output:
            kernel_source += '''
                    if(is_first_work_item){
                        if(i % ''' + str(thinning) + ''' == 0){
    
                            data->_log_likelihoods[i / ''' + str(thinning) + '''] = current_likelihood;
                            data->_log_priors[i / ''' + str(thinning) + '''] = current_prior;
    
                            for(uint j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                                data->_samples[(ulong)(i / ''' + str(thinning) + ''') // remove the interval
                                        + j * ''' + str(nmr_samples) + '''  // parameter index
                                ] = current_position[j];
                            }
                        }
                    }
        '''
        kernel_source += '''
                    _advanceSampler(data, i + data->_iteration_offset, rng_data, 
                                    current_position, &current_likelihood, &current_prior, data->_log_likelihood_tmp);
                }

                if(is_first_work_item){
                    for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        data->_current_chain_position[i] = current_position[i];
                    }

                    uint state[8];
                    rand123_data_to_array(rand123_rng_data, state);
                    for(uint i = 0; i < 6; i++){
                        data->_rng_state[i] = state[i];
                    }
                }
            }
        '''
        return NameFunctionTuple('compute', kernel_source)

    def _get_state_update_cl_func(self, nmr_samples, thinning, return_output):
        """Get the function that can advance the sampler state.

        This function is called by the MCMC sampler to draw and return a new sample.

        Args:
            nmr_samples (int): the number of samples we will draw
            thinning (int): the thinning factor we want to use
            return_output (boolean): if the kernel should return output

        Returns:
            str: a CL function with signature:

            .. code-block:: c
                void _advanceSampler(mot_data_struct* data,
                                     ulong current_iteration,
                                     void* rng_data,
                                     local mot_float_type* current_position,
                                     local double* const current_likelihood,
                                     local mot_float_type* const current_prior,
                                     local double* log_likelihood_tmp);
        """
        raise NotImplementedError()

    def _get_log_prior_cl_func(self):
        """Get the CL log prior compute function.

        Returns:
            str: the compute function for computing the log prior of the model.
        """
        prior_func = self._model.get_log_prior_function(address_space_parameter_vector='local')
        kernel_source = prior_func.get_cl_code()
        kernel_source += '''
            mot_float_type _computeLogPrior(mot_data_struct* data,
                                            local const mot_float_type* const x){
                return ''' + prior_func.get_cl_function_name() + '''(data, x);
            }
        '''
        return kernel_source

    def _get_log_likelihood_cl_func(self):
        """Get the CL log likelihood compute function.

        This uses local reduction to compute the log likelihood for every observation in CL local space.
        The results are then summed in the first work item and returned using a pointer.

        Returns:
            str: the CL code for the log likelihood compute func.
        """
        ll_func = self._model.get_log_likelihood_per_observation_function()

        kernel_source = ll_func.get_cl_code()
        kernel_source += '''
            void _computeLogLikelihood(mot_data_struct* data,
                                          local mot_float_type* const current_position,
                                          local double* log_likelihood_tmp,
                                          local double* likelihood_sum){

                ulong observation_ind;
                ulong local_id = get_local_id(0);
                log_likelihood_tmp[local_id] = 0;
                uint workgroup_size = get_local_size(0);
                uint elements_for_workitem = ceil(NMR_OBSERVATIONS / (mot_float_type)workgroup_size);

                if(workgroup_size * (elements_for_workitem - 1) + local_id >= NMR_OBSERVATIONS){
                    elements_for_workitem -= 1;
                }

                mot_float_type x_private[''' + str(self._nmr_params) + '''];
                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    x_private[i] = current_position[i];
                }

                for(uint i = 0; i < elements_for_workitem; i++){
                    observation_ind = i * workgroup_size + local_id;

                    log_likelihood_tmp[local_id] += ''' + ll_func.get_cl_function_name() + '''(
                        data, x_private, observation_ind);
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                if(get_local_id(0) == 0){    
                    *likelihood_sum = 0;
                    for(uint i = 0; i < get_local_size(0); i++){
                        *likelihood_sum += log_likelihood_tmp[i];
                    }
                }
            }
        '''
        return kernel_source

    @contextmanager
    def _logging(self, nmr_samples, burnin, thinning):
        self._logger.info('Starting sampling with method {0}'.format(self.__class__.__name__))
        self._logger.info('We will use a {} precision float type for the calculations.'.format(
            'double' if self._cl_runtime_info.double_precision else 'single'))

        for env in self._cl_runtime_info.get_cl_environments():
            self._logger.info('Using device \'{}\'.'.format(str(env)))

        self._logger.info('Using compile flags: {}'.format(self._cl_runtime_info.get_compile_flags()))

        sample_settings = dict(nmr_samples=nmr_samples,
                               burnin=burnin,
                               thinning=thinning)
        self._logger.info('Sample settings: nmr_samples: {nmr_samples}, burnin: {burnin}, '
                          'thinning: {thinning}. '.format(**sample_settings))

        samples_drawn = dict(samples_drawn=(burnin + thinning * nmr_samples),
                             samples_returned=nmr_samples)
        self._logger.info('Total samples drawn: {samples_drawn}, total samples returned: '
                          '{samples_returned} (per problem).'.format(**samples_drawn))
        yield
        self._logger.info('Finished sampling')


class AbstractRWMSampler(AbstractSampler):

    def __init__(self, model, starting_positions, proposal_stds, use_random_scan=False, **kwargs):
        """An abstract basis for Random Walk Metropolis (RWM) samplers.

        Random Walk Metropolis (RWM) samplers require for every parameter and every modeling instance an proposal
        standard deviation, used in the random walk.

        Args:
            model (SampleModelInterface): the model to sample.
            starting_positions (ndarray): the starting positions for the sampler. Should be a two dimensional matrix
                with for every modeling instance (first dimension) and every parameter (second dimension) a value.
            proposal_stds (ndarray): for every parameter and every modeling instance an initial proposal std.
            use_random_scan (boolean): if we iterate over the parameters in a random order or in a linear order
                at every sample iteration. By default we apply a system scan starting from the first dimension to the
                last. With a random scan we randomize the indices every iteration.
        """
        super(AbstractRWMSampler, self).__init__(model, starting_positions, **kwargs)
        self._proposal_stds = np.copy(np.require(proposal_stds, requirements='CAOW',
                                                 dtype=self._cl_runtime_info.mot_float_dtype))
        self._use_random_scan = use_random_scan

    def _get_kernel_data(self, nmr_samples, thinning, return_output):
        kernel_data = super(AbstractRWMSampler, self)._get_kernel_data(nmr_samples, thinning, return_output)
        kernel_data.update({
            '_proposal_stds': KernelArray(self._proposal_stds, 'mot_float_type', is_writable=True,
                                          ensure_zero_copy=True)
        })
        return kernel_data

    def _get_proposal_update_function(self, nmr_samples, thinning, return_output):
        """Get the proposal update function.

        Returns:
            str: the update function for the proposal standard deviations.
                Should be a string with CL code with the signature:

                .. code-block:: c
                    void _updateProposalState(mot_data_struct* data,
                                              ulong current_iteration,
                                              local mot_float_type* current_position);
        """
        return '''
            void _updateProposalState(mot_data_struct* data, ulong current_iteration){}
        '''

    def _at_acceptance_callback_c_func(self):
        """Get a CL function that is to be applied at the moment a sample is accepted.

        Returns:
            str: a piece of C code to be applied when the function is accepted.
                Should be a string with CL code with the signature:

                .. code-block:: c
                    void _sampleAccepted(mot_data_struct* data,
                                         ulong current_iteration,
                                         uint parameter_ind);
        """
        return '''
            void _sampleAccepted(mot_data_struct* data, ulong current_iteration, uint parameter_ind){}
        '''

    def _get_state_update_cl_func(self, nmr_samples, thinning, return_output):
        proposal_finalize_func = self._model.get_finalize_proposal_function('local')

        kernel_source = self._get_proposal_update_function(nmr_samples, thinning, return_output)
        kernel_source += self._at_acceptance_callback_c_func()
        kernel_source += proposal_finalize_func.get_cl_code()

        if self._use_random_scan:
            kernel_source += '''
                void _shuffle(uint* array, uint n, void* rng_data){
                    if(n > 1){
                        for(uint i = 0; i < n - 1; i++){
                          uint j = (uint)(frand(rng_data) * (n - i) + i); 
                          
                          uint tmp = array[j];
                          array[j] = array[i];
                          array[i] = tmp;
                        }
                    }
                }
            '''

        kernel_source += '''
            void _advanceSampler(
                    mot_data_struct* data,
                    ulong current_iteration, 
                    void* rng_data,
                    local mot_float_type* current_position,
                    local double* const current_likelihood,
                    local mot_float_type* const current_prior,
                    local double* log_likelihood_tmp){

                local mot_float_type new_position[''' + str(self._nmr_params) + '''];
                local mot_float_type new_prior;
                local double new_likelihood;
                local double bayesian_f;
                bool is_first_work_item = get_local_id(0) == 0;

                if(is_first_work_item){
                    for(uint k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                        new_position[k] = current_position[k];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);     
        '''
        if self._use_random_scan:
            kernel_source += '''
                uint indices[] = {''' + ', '.join(map(str, range(self._nmr_params))) + '''};
                _shuffle(indices, ''' + str(self._nmr_params) + ''', rng_data);
                
                for(uint ind = 0; ind < ''' + str(self._nmr_params) + '''; ind++){
                    uint k = indices[ind];    
            '''
        else:
            kernel_source += 'for(uint k = 0; k < ' + str(self._nmr_params) + '; k++){'
        kernel_source += '''
                    if(is_first_work_item){
                        new_position[k] += frandn(rng_data) * data->_proposal_stds[k];
                        ''' + proposal_finalize_func.get_cl_function_name() + '''(data, new_position);
                        new_prior = _computeLogPrior(data, new_position);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    if(exp(new_prior) > 0){
                        _computeLogLikelihood(data, new_position, log_likelihood_tmp, &new_likelihood);

                        if(is_first_work_item){
                            bayesian_f = exp((new_likelihood + new_prior) - (*current_likelihood + *current_prior));

                            if(frand(rng_data) < bayesian_f){
                                *current_likelihood = new_likelihood;
                                *current_prior = new_prior;
                                _sampleAccepted(data, current_iteration, k);
                            }
                            else{
                                new_position[k] = current_position[k];
                            }
                        }
                    }
                    else{ // prior returned 0
                        if(is_first_work_item){
                            new_position[k] = current_position[k];
                        }
                    }
                }
                
                if(is_first_work_item){
                    for(uint k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                        current_position[k] = new_position[k];
                    }    
                    _updateProposalState(data, current_iteration, current_position);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        '''
        return kernel_source


class SamplingOutput(object):

    def get_samples(self):
        """Get the matrix containing the sampling results.

        Returns:
            ndarray: the sampled parameter maps, a (d, p, n) array with for d problems and p parameters n samples.
        """
        raise NotImplementedError()

    def get_log_likelihoods(self):
        """Get per set of sampled parameters the log likelihood value associated with that set of parameters.

        Returns:
            ndarray: the log likelihood values, a (d, n) array with for d problems and n samples the log likelihood
                value.
        """
        raise NotImplementedError()

    def get_log_priors(self):
        """Get per set of sampled parameters the log prior value associated with that set of parameters.

        Returns:
            ndarray: the log prior values, a (d, n) array with for d problems and n samples the prior value.
        """
        raise NotImplementedError()


class SimpleSampleOutput(SamplingOutput):

    def __init__(self, samples, log_likelihoods, log_priors):
        """Simple storage container for the sampling output"""
        self._samples = samples
        self._log_likelihood = log_likelihoods
        self._log_prior = log_priors

    def get_samples(self):
        return self._samples

    def get_log_likelihoods(self):
        return self._log_likelihood

    def get_log_priors(self):
        return self._log_prior
