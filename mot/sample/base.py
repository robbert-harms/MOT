import logging
from contextlib import contextmanager

from mot.lib.cl_function import SimpleCLFunction, SimpleCLCodeObject
from mot.configuration import CLRuntimeInfo
from mot.library_functions import Rand123
from mot.lib.utils import split_in_batches
from mot.lib.kernel_data import Scalar, Array, \
    Zeros, Struct, LocalMemory
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2014-05-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractSampler:

    def __init__(self, ll_func, log_prior_func, x0, data=None, cl_runtime_info=None, **kwargs):
        """Abstract base class for sample routines.

        Sampling routines implementing this interface should be stateful objects that, for the given likelihood
        and prior, keep track of the sample state over multiple calls to :meth:`sample`.

        Args:
            ll_func (mot.lib.cl_function.CLFunction): The log-likelihood function. A CL function with the signature:

                .. code-block:: c

                        double <func_name>(local const mot_float_type* const x, void* data);

            log_prior_func (mot.lib.cl_function.CLFunction): The log-prior function. A CL function with the signature:

                .. code-block:: c

                    mot_float_type <func_name>(local const mot_float_type* const x, void* data);

            x0 (ndarray): the starting positions for the sampler. Should be a two dimensional matrix
                with for every modeling instance (first dimension) and every parameter (second dimension) a value.
            data (mot.lib.kernel_data.KernelData): the user provided data for the ``void* data`` pointer.
        """
        self._cl_runtime_info = cl_runtime_info or CLRuntimeInfo()
        self._logger = logging.getLogger(__name__)
        self._ll_func = ll_func
        self._log_prior_func = log_prior_func
        self._data = data
        self._x0 = x0
        if len(x0.shape) < 2:
            self._x0 = self._x0[..., None]
        self._nmr_problems = self._x0.shape[0]
        self._nmr_params = self._x0.shape[1]
        self._sampling_index = 0

        float_type = self._cl_runtime_info.mot_float_dtype
        self._current_chain_position = np.require(np.copy(self._x0), requirements='CAOW', dtype=float_type)
        self._current_log_likelihood = np.zeros(self._nmr_problems, dtype=float_type)
        self._current_log_prior = np.zeros(self._nmr_problems, dtype=float_type)
        self._rng_state = np.random.uniform(low=np.iinfo(np.uint32).min, high=np.iinfo(np.uint32).max + 1,
                                            size=(self._nmr_problems, 6)).astype(np.uint32)

        self._initialize_likelihood_prior(self._current_chain_position, self._current_log_likelihood,
                                          self._current_log_prior)

    def _get_mcmc_method_kernel_data(self):
        """Get the kernel data specific for the implemented method.

        This will be provided as a void pointer to the implementing MCMC method.

        Returns:
            mot.lib.kernel_data.KernelData: the kernel data object
        """
        raise NotImplementedError()

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
                void _advanceSampler(void* method_data,
                                     void* data,
                                     ulong current_iteration,
                                     void* rng_data,
                                     global mot_float_type* current_position,
                                     global mot_float_type* current_log_likelihood,
                                     global mot_float_type* current_log_prior);
        """
        raise NotImplementedError()

    def set_cl_runtime_info(self, cl_runtime_info):
        """Update the CL runtime information.

        Args:
            cl_runtime_info (mot.configuration.CLRuntimeInfo): the new runtime information
        """
        self._cl_runtime_info = cl_runtime_info

    def sample(self, nmr_samples, burnin=0, thinning=1):
        """Take additional samples from the given likelihood and prior, using this sampler.

        This method can be called multiple times in which the sample state is stored in between.

        Args:
            nmr_samples (int): the number of samples to return
            burnin (int): the number of samples to discard before returning samples
            thinning (int): how many sample we wait before storing a new one. This will draw extra samples such that
                    the total number of samples generated is ``nmr_samples * (thinning)`` and the number of samples
                    stored is ``nmr_samples``. If set to one or lower we store every sample after the burn in.

        Returns:
            SamplingOutput: the sample output object
        """
        if not thinning or thinning < 1:
            thinning = 1
        if not burnin or burnin < 0:
            burnin = 0

        max_samples_per_batch = max(1000 // thinning, 100)

        with self._logging(nmr_samples, burnin, thinning):
            if burnin > 0:
                for batch_start, batch_end in split_in_batches(burnin, max_samples_per_batch):
                    self._sample(batch_end - batch_start, return_output=False)
            if nmr_samples > 0:
                outputs = []
                for batch_start, batch_end in split_in_batches(nmr_samples, max_samples_per_batch):
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
        sample_func = self._get_compute_func(nmr_samples, thinning, return_output)
        sample_func.evaluate(kernel_data, self._nmr_problems,
                             use_local_reduction=all(env.is_gpu for env in self._cl_runtime_info.cl_environments),
                             cl_runtime_info=self._cl_runtime_info)
        self._sampling_index += nmr_samples * thinning
        if return_output:
            return (kernel_data['samples'].get_data(),
                    kernel_data['log_likelihoods'].get_data(),
                    kernel_data['log_priors'].get_data())

    def _initialize_likelihood_prior(self, positions, log_likelihoods, log_priors):
        """Initialize the likelihood and the prior using the given positions.

        This is a general method for computing the log likelihoods and log priors for given positions.

        Subclasses can use this to instantiate secondary chains as well.
        """
        func = SimpleCLFunction.from_string('''
            void compute(global mot_float_type* chain_position,
                         global mot_float_type* log_likelihood,
                         global mot_float_type* log_prior,
                         local mot_float_type* x_tmp,
                         void* data){
                
                bool is_first_work_item = get_local_id(0) == 0;
    
                if(is_first_work_item){
                    for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x_tmp[i] = chain_position[i];
                    }
                    *log_prior = _computeLogPrior(x_tmp, data);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                    
                *log_likelihood = _computeLogLikelihood(x_tmp, data);
            }
        ''', dependencies=[self._get_log_prior_cl_func(), self._get_log_likelihood_cl_func()])

        kernel_data = {
            'chain_position': Array(positions, 'mot_float_type', mode='rw', ensure_zero_copy=True),
            'log_likelihood': Array(log_likelihoods, 'mot_float_type', mode='rw', ensure_zero_copy=True),
            'log_prior': Array(log_priors, 'mot_float_type', mode='rw', ensure_zero_copy=True),
            'x_tmp': LocalMemory('mot_float_type', self._nmr_params),
            'data': self._data
        }

        func.evaluate(kernel_data, self._nmr_problems,
                      use_local_reduction=all(env.is_gpu for env in self._cl_runtime_info.cl_environments),
                      cl_runtime_info=self._cl_runtime_info)

    def _get_kernel_data(self, nmr_samples, thinning, return_output):
        """Get the kernel data we will input to the MCMC sampler.

        This sets the items:

        * data: the pointer to the user provided data
        * method_data: the data specific to the MCMC method
        * nmr_iterations: the number of iterations to sample
        * iteration_offset: the current sample index, that is, the offset to the given number of iterations
        * rng_state: the random number generator state
        * current_chain_position: the current position of the sampled chain
        * current_log_likelihood: the log likelihood of the current position on the chain
        * current_log_prior: the log prior of the current position on the chain

        Additionally, if ``return_output`` is True, we add to that the arrays:

        * samples: for the samples
        * log_likelihoods: for storing the log likelihoods
        * log_priors: for storing the priors

        Args:
            nmr_samples (int): the number of samples we will draw
            thinning (int): the thinning factor we want to use
            return_output (boolean): if the kernel should return output

        Returns:
            dict[str: mot.lib.utils.KernelData]: the kernel input data
        """
        kernel_data = {
            'data': self._data,
            'method_data': self._get_mcmc_method_kernel_data(),
            'nmr_iterations': Scalar(nmr_samples * thinning, ctype='ulong'),
            'iteration_offset': Scalar(self._sampling_index, ctype='ulong'),
            'rng_state': Array(self._rng_state, 'uint', mode='rw', ensure_zero_copy=True),
            'current_chain_position': Array(self._current_chain_position, 'mot_float_type',
                                            mode='rw', ensure_zero_copy=True),
            'current_log_likelihood': Array(self._current_log_likelihood, 'mot_float_type',
                                            mode='rw', ensure_zero_copy=True),
            'current_log_prior': Array(self._current_log_prior, 'mot_float_type',
                                       mode='rw', ensure_zero_copy=True),
        }

        if return_output:
            kernel_data.update({
                'samples': Zeros((self._nmr_problems, self._nmr_params, nmr_samples), ctype='mot_float_type'),
                'log_likelihoods': Zeros((self._nmr_problems, nmr_samples), ctype='mot_float_type'),
                'log_priors': Zeros((self._nmr_problems, nmr_samples), ctype='mot_float_type'),
            })
        return kernel_data

    def _get_compute_func(self, nmr_samples, thinning, return_output):
        """Get the MCMC algorithm as a computable function.

        Args:
            nmr_samples (int): the number of samples we will draw
            thinning (int): the thinning factor we want to use
            return_output (boolean): if the kernel should return output

        Returns:
            mot.lib.cl_function.CLFunction: the compute function
        """
        cl_func = '''
            void compute(global uint* rng_state, 
                         global mot_float_type* current_chain_position,
                         global mot_float_type* current_log_likelihood,
                         global mot_float_type* current_log_prior,
                         ulong iteration_offset, 
                         ulong nmr_iterations, 
                         ''' + ('''global mot_float_type* samples, 
                                   global mot_float_type* log_likelihoods,
                                   global mot_float_type* log_priors,''' if return_output else '') + '''
                         void* method_data, 
                         void* data){
                         
                bool is_first_work_item = get_local_id(0) == 0;
    
                rand123_data rand123_rng_data = rand123_initialize_data((uint[]){
                    rng_state[0], rng_state[1], rng_state[2], rng_state[3], 
                    rng_state[4], rng_state[5], 0, 0});
                void* rng_data = (void*)&rand123_rng_data;
        
                for(ulong i = 0; i < nmr_iterations; i++){
        '''
        if return_output:
            cl_func += '''
                    if(is_first_work_item){
                        if(i % ''' + str(thinning) + ''' == 0){
                            log_likelihoods[i / ''' + str(thinning) + '''] = *current_log_likelihood;
                            log_priors[i / ''' + str(thinning) + '''] = *current_log_prior;
    
                            for(uint j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                                samples[(ulong)(i / ''' + str(thinning) + ''') // remove the interval
                                        + j * ''' + str(nmr_samples) + '''  // parameter index
                                ] = current_chain_position[j];
                            }
                        }
                    }
        '''
        cl_func += '''
                    _advanceSampler(method_data, data, i + iteration_offset, rng_data, 
                                    current_chain_position, current_log_likelihood, current_log_prior);
                }

                if(is_first_work_item){
                    uint state[8];
                    rand123_data_to_array(rand123_rng_data, state);
                    for(uint i = 0; i < 6; i++){
                        rng_state[i] = state[i];
                    }
                }
            }
        '''
        return SimpleCLFunction.from_string(
            cl_func,
            dependencies=[Rand123(), self._get_log_prior_cl_func(),
                          self._get_log_likelihood_cl_func(),
                          SimpleCLCodeObject(self._get_state_update_cl_func(nmr_samples, thinning, return_output))])

    def _get_log_prior_cl_func(self):
        """Get the CL log prior compute function.

        Returns:
            str: the compute function for computing the log prior.
        """
        return SimpleCLFunction.from_string('''
            mot_float_type _computeLogPrior(local const mot_float_type* x, void* data){
                return ''' + self._log_prior_func.get_cl_function_name() + '''(x, data);
            }
        ''', dependencies=[self._log_prior_func])

    def _get_log_likelihood_cl_func(self):
        """Get the CL log likelihood compute function.

        This uses local reduction to compute the log likelihood for every observation in CL local space.
        The results are then summed in the first work item and returned using a pointer.

        Returns:
            str: the CL code for the log likelihood compute func.
        """
        return SimpleCLFunction.from_string('''
            double _computeLogLikelihood(local const mot_float_type* current_position, void* data){
                return ''' + self._ll_func.get_cl_function_name() + '''(current_position, data);
            }
        ''', dependencies=[self._ll_func])

    @contextmanager
    def _logging(self, nmr_samples, burnin, thinning):
        self._logger.info('Starting sample with method {0}'.format(self.__class__.__name__))
        self._logger.info('We will use a {} precision float type for the calculations.'.format(
            'double' if self._cl_runtime_info.double_precision else 'single'))

        for env in self._cl_runtime_info.cl_environments:
            self._logger.info('Using device \'{}\'.'.format(str(env)))

        self._logger.info('Using compile flags: {}'.format(self._cl_runtime_info.compile_flags))

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
        self._logger.info('Finished sample')


class AbstractRWMSampler(AbstractSampler):

    def __init__(self, ll_func, log_prior_func, x0, proposal_stds, use_random_scan=False,
                 finalize_proposal_func=None, **kwargs):
        """An abstract basis for Random Walk Metropolis (RWM) samplers.

        Random Walk Metropolis (RWM) samplers require for every parameter and every modeling instance an proposal
        standard deviation, used in the random walk.

        Args:
            ll_func (mot.lib.cl_function.CLFunction): The log-likelihood function.
            log_prior_func (mot.lib.cl_function.CLFunction): The log-prior function.
            x0 (ndarray): the starting positions for the sampler. Should be a two dimensional matrix
                with for every modeling instance (first dimension) and every parameter (second dimension) a value.
            proposal_stds (ndarray): for every parameter and every modeling instance an initial proposal std.
            use_random_scan (boolean): if we iterate over the parameters in a random order or in a linear order
                at every sample iteration. By default we apply a system scan starting from the first dimension to the
                last. With a random scan we randomize the indices every iteration.
            finalize_proposal_func (mot.lib.cl_function.CLFunction): a CL function to finalize every proposal by
                the sampling routine. This allows the model to change a proposal before computing the
                prior or likelihood probabilities. If None, we will not use this callback.

                As an example, suppose you are sampling a polar coordinate :math:`\theta` defined on
                :math:`[0, 2\pi]` with a random walk Metropolis proposal distribution. This distribution might propose
                positions outside of the range of :math:`\theta`. Of course the model function could deal with that by
                taking the modulus of the input, but then you have to post-process the chain with the same
                transformation. Instead, this function allows changing the proposal before it is put into the model
                and before it will be stored in the chain.

                This function should return a proposal that is equivalent (but not necessarily equal)
                to the provided proposal.

                Signature:

                .. code-block:: c

                    void <func_name>(void* data, local mot_float_type* x);
        """
        super().__init__(ll_func, log_prior_func, x0, **kwargs)
        self._proposal_stds = np.require(np.copy(proposal_stds), requirements='CAOW',
                                         dtype=self._cl_runtime_info.mot_float_dtype)
        self._use_random_scan = use_random_scan
        self._finalize_proposal_func = finalize_proposal_func or SimpleCLFunction.from_string(
            'void finalizeProposal(void* data, local mot_float_type* x){}')

    def _get_mcmc_method_kernel_data(self):
        return Struct(self._get_mcmc_method_kernel_data_elements(), '_mcmc_method_data')

    def _get_mcmc_method_kernel_data_elements(self):
        """Get the mcmc method kernel data elements. Used by :meth:`_get_mcmc_method_kernel_data`."""
        return {'proposal_stds': Array(self._proposal_stds, 'mot_float_type', mode='rw', ensure_zero_copy=True),
                'x_tmp': LocalMemory('mot_float_type', nmr_items=1 + self._nmr_params)}

    def _get_proposal_update_function(self, nmr_samples, thinning, return_output):
        """Get the proposal update function.

        Returns:
            str: the update function for the proposal standard deviations.
                Should be a string with CL code with the signature:

                .. code-block:: c
                    void _updateProposalState(_mcmc_method_data* method_data,
                                              ulong current_iteration,
                                              global mot_float_type* current_position);
        """
        return '''
            void _updateProposalState(_mcmc_method_data* method_data, ulong current_iteration,
                                      global mot_float_type* current_position){}
        '''

    def _at_acceptance_callback_c_func(self):
        """Get a CL function that is to be applied at the moment a sample is accepted.

        Returns:
            str: a piece of C code to be applied when the function is accepted.
                Should be a string with CL code with the signature:

                .. code-block:: c
                    void _sampleAccepted(_mcmc_method_data* method_data,
                                         ulong current_iteration,
                                         uint parameter_ind);
        """
        return '''
            void _sampleAccepted(_mcmc_method_data* method_data, ulong current_iteration, uint parameter_ind){}
        '''

    def _get_state_update_cl_func(self, nmr_samples, thinning, return_output):
        kernel_source = self._get_proposal_update_function(nmr_samples, thinning, return_output)
        kernel_source += self._at_acceptance_callback_c_func()
        kernel_source += self._finalize_proposal_func.get_cl_code()

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
                    void* method_data,
                    void* data,
                    ulong current_iteration, 
                    void* rng_data,
                    global mot_float_type* current_position,
                    global mot_float_type* current_log_likelihood,
                    global mot_float_type* current_log_prior){
                
                local mot_float_type* new_log_prior = ((_mcmc_method_data*)method_data)->x_tmp;
                *new_log_prior = 0;
                local mot_float_type* new_position = ((_mcmc_method_data*)method_data)->x_tmp + 1;
                
                mot_float_type new_log_likelihood;
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
                        new_position[k] += frandn(rng_data) * ((_mcmc_method_data*)method_data)->proposal_stds[k];
                        ''' + self._finalize_proposal_func.get_cl_function_name() + '''(data, new_position);
                        *new_log_prior = _computeLogPrior(new_position, data);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    if(exp(*new_log_prior) > 0){
                        new_log_likelihood = _computeLogLikelihood(new_position, data);
                        
                        if(is_first_work_item){
                            if(frand(rng_data) < exp((new_log_likelihood + *new_log_prior) 
                                                     - (*current_log_likelihood + *current_log_prior))){
                                *current_log_likelihood = new_log_likelihood;
                                *current_log_prior = *new_log_prior;
                                for(uint k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                                    current_position[k] = new_position[k];
                                }           
                                _sampleAccepted((_mcmc_method_data*)method_data, current_iteration, k);
                            }
                            else{
                                for(uint k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                                    new_position[k] = current_position[k];
                                }
                            }
                        }
                    }
                    else{ // prior returned 0
                        if(is_first_work_item){
                            for(uint k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                                new_position[k] = current_position[k];
                            }
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if(is_first_work_item){
                    _updateProposalState((_mcmc_method_data*)method_data, current_iteration, current_position);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        '''
        return kernel_source


class SamplingOutput:

    def get_samples(self):
        """Get the matrix containing the sample results.

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
        """Simple storage container for the sample output"""
        self._samples = samples
        self._log_likelihood = log_likelihoods
        self._log_prior = log_priors

    def get_samples(self):
        return self._samples

    def get_log_likelihoods(self):
        return self._log_likelihood

    def get_log_priors(self):
        return self._log_prior
