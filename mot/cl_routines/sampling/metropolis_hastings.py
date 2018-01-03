import warnings
import numpy as np
import pyopencl as cl
from mot.library_functions import Rand123
from ...cl_routines.sampling.base import AbstractSampler, SimpleSampleOutput
from ...load_balance_strategies import Worker
from ...utils import get_float_type_def, KernelInputDataManager, split_in_batches

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MetropolisHastings(AbstractSampler):

    def __init__(self, nmr_samples=None, burn_length=None,
                 sample_intervals=None, use_adaptive_proposals=True, **kwargs):
        """An CL implementation of Metropolis Hastings.

        This implementation uses a random walk single component updating strategy for the sampling.

        Args:
            nmr_samples (int): The length of the (returned) chain per voxel, defaults to 1
            burn_length (int): The length of the burn in (per voxel), these are extra samples,
                during burn-in we don't apply the thinning.
            sample_intervals (int): how many sample we wait before storing a new one.
                This will draw extra samples such that the total number of samples generated is
                ``chain_length * (sample_intervals + 1)`` and the number of samples stored is
                ``chain_length``. If set to zero we store every sample after the burn in.
            use_adaptive_proposals (boolean): if we use the adaptive proposals (set to True) or not (set to False).
        """
        super(MetropolisHastings, self).__init__(**kwargs)
        self._nmr_samples = nmr_samples or 1000
        self.burn_length = burn_length or 0
        self.sample_intervals = sample_intervals or 0
        self.use_adaptive_proposals = use_adaptive_proposals

        if self.burn_length is None:
            self.burn_length = 0

        if self.sample_intervals is None:
            self.sample_intervals = 0

        if self.burn_length < 0:
            raise ValueError('The burn length can not be smaller than 0, {} given'.format(self.burn_length))
        if self.sample_intervals < 0:
            raise ValueError('The sampling interval can not be smaller than 0, {} given'.format(self.sample_intervals))
        if self._nmr_samples < 1:
            raise ValueError('The number of samples to draw can '
                             'not be smaller than 1, {} given'.format(self._nmr_samples))

    @property
    def nmr_samples(self):
        return self._nmr_samples

    def sample(self, model, init_params=None):
        """Sample the given model with Metropolis Hastings

        Returns:
            MHSampleOutput: extension of the default output with some more data
        """
        mot_float_dtype = np.float32
        if model.double_precision:
            mot_float_dtype = np.float64

        self._do_initial_logging(model)

        if init_params is None:
            init_params = model.get_initial_parameters()

        nmr_params = init_params.shape[1]

        current_chain_position = np.require(init_params, mot_float_dtype, requirements=['C', 'A', 'O', 'W'])
        proposal_state = np.require(model.get_proposal_state(), mot_float_dtype, requirements=['C', 'A', 'O', 'W'])
        mh_state = _prepare_mh_state(model.get_metropolis_hastings_state(), mot_float_dtype)

        samples = np.zeros((model.get_nmr_problems(), nmr_params, self.nmr_samples),
                           dtype=mot_float_dtype, order='C')
        log_likelihoods = np.zeros((model.get_nmr_problems(), self.nmr_samples), dtype=mot_float_dtype, order='C')
        log_priors = np.zeros((model.get_nmr_problems(), self.nmr_samples), dtype=mot_float_dtype, order='C')

        def run(_samples, _log_likelihoods, _log_priors, _mh_state, nmr_samples, in_burnin=False):
            """Create the worker, process it, store the results in the given sample array and return a new mh state."""
            if in_burnin:
                sample_interval = 0
            else:
                sample_interval = self.sample_intervals

            workers = self._create_workers(lambda cl_environment: _MHWorker(
                cl_environment, self.get_compile_flags_list(model.double_precision), model, current_chain_position,
                _samples, _log_likelihoods, _log_priors, proposal_state, _mh_state, nmr_samples, in_burnin,
                sample_interval, self.use_adaptive_proposals, mot_float_dtype))
            self.load_balancer.process(workers, model.get_nmr_problems())
            return _mh_state.with_nmr_samples_drawn(_mh_state.nmr_samples_drawn + nmr_samples * (sample_interval + 1))

        self._logger.info('Starting sampling with method {0}'.format(self.__class__.__name__))

        if self.burn_length > 0:
            mh_state = run(samples, None, None, mh_state, self.burn_length, in_burnin=True)

        for batch_start, batch_end in split_in_batches(self.nmr_samples, 1000):
            batch_size = batch_end - batch_start

            samples_subset = np.zeros((model.get_nmr_problems(), nmr_params, batch_size),
                                      dtype=mot_float_dtype, order='C')
            ll_subset = np.zeros((model.get_nmr_problems(), batch_size), dtype=mot_float_dtype, order='C')
            lp_subset = np.zeros((model.get_nmr_problems(), batch_size), dtype=mot_float_dtype, order='C')

            mh_state = run(samples_subset, ll_subset, lp_subset, mh_state, batch_size)

            samples[..., batch_start:batch_end] = samples_subset
            log_likelihoods[..., batch_start:batch_end] = ll_subset
            log_priors[..., batch_start:batch_end] = lp_subset

        self._logger.info('Finished sampling')

        return MHSampleOutput(samples, log_likelihoods, log_priors, proposal_state, mh_state, current_chain_position)

    def _do_initial_logging(self, model):
        self._logger.info('Entered sampling routine.')
        self._logger.info('We will use a {} precision float type for the calculations.'.format(
            'double' if model.double_precision else 'single'))

        for env in self.load_balancer.get_used_cl_environments(self.cl_environments):
            self._logger.info('Using device \'{}\'.'.format(str(env)))

        self._logger.info('Using compile flags: {}'.format(self.get_compile_flags_list(model.double_precision)))

        sample_settings = dict(nmr_samples=self.nmr_samples,
                               burn_length=self.burn_length,
                               sample_intervals=self.sample_intervals,
                               use_adaptive_proposals=self.use_adaptive_proposals)
        self._logger.info('Sample settings: nmr_samples: {nmr_samples}, burn_length: {burn_length}, '
                          'sample_intervals: {sample_intervals}, '
                          'use_adaptive_proposals: {use_adaptive_proposals}. '.format(**sample_settings))

        samples_drawn = dict(samples_drawn=(self.burn_length + (self.sample_intervals + 1) * self.nmr_samples),
                             samples_returned=self.nmr_samples)
        self._logger.info('Total samples drawn: {samples_drawn}, total samples returned: '
                          '{samples_returned} (per problem).'.format(**samples_drawn))


class MHSampleOutput(SimpleSampleOutput):

    def __init__(self, samples, log_likelihoods, log_priors, proposal_state, mh_state, current_chain_position):
        """Simple storage container for the sampling output

        Args:
            samples (ndarray): an (d, p, n) matrix with d problems, p parameters and n samples
            log_likelihoods (ndarray): the log likelihood values, a (d, n) array with for d problems and n samples the
                log likelihood value.
            log_priors (ndarray): the log prior values, a (d, n) array with for d problems and n samples the
                prior value.
            proposal_state (ndarray): (d, p) matrix with for d problems and p parameters the proposal state
            mh_state (MHState): the current MH state
            current_chain_position (ndarray): (d, p) matrix with for d observations and p parameters the current
                chain position. If the samples are not empty the last element in the samples (``samples[..., -1]``)
                should equal this matrix.
        """
        super(MHSampleOutput, self).__init__(samples, log_likelihoods, log_priors)
        self._proposal_state = proposal_state
        self._mh_state = mh_state
        self._current_chain_position = current_chain_position

    def get_proposal_state(self):
        """Get the proposal state at the end of this sampling

        Returns:
            ndarray: a (d, p) array with for d problems and p parameters the proposal state
        """
        return self._proposal_state

    def get_mh_state(self):
        """Get the Metropolis Hastings state as it was at the end of this sampling run.

        Returns:
            MHState: the current MH state
        """
        return self._mh_state

    def get_current_chain_position(self):
        """Get the current chain position current_chain_position

        Returns:
            ndarray: (d, p) matrix with for d observations and p parameters the current
                chain position. If the samples are not empty the last element in the samples (``samples[..., -1]``)
                should equal this matrix.
        """
        return self._current_chain_position


class _MHWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, current_chain_position, samples,
                 log_likelihoods, log_priors, proposal_state, mh_state, nmr_samples, in_burnin,
                 sample_intervals, use_adaptive_proposals, mot_float_dtype):
        super(_MHWorker, self).__init__(cl_environment)

        self._model = model
        self._data_info = self._model.get_kernel_data()
        self._data_struct_manager = KernelInputDataManager(self._data_info, mot_float_dtype)
        self._current_chain_position = current_chain_position
        self._nmr_params = current_chain_position.shape[1]
        self._samples = samples
        self._log_likelihoods = log_likelihoods
        self._log_priors = log_priors

        self._proposal_state = proposal_state
        self._mh_state = mh_state
        self._mh_state_dict = self._get_mh_state_dict()

        self._nmr_samples = nmr_samples
        self._in_burnin = in_burnin
        self._sample_intervals = sample_intervals

        kernel_builder = _MCMCKernelBuilder(
            compile_flags, self._mh_state_dict, cl_environment, model,
            nmr_samples, not in_burnin, sample_intervals, use_adaptive_proposals, self._nmr_params, mot_float_dtype)

        self._kernel = kernel_builder.build()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        workgroup_size = cl.Kernel(self._kernel, 'sample').get_work_group_info(
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, self._cl_environment.device)

        data_buffers, readout_items = self._get_buffers(workgroup_size)

        kernel_func = self._kernel.sample
        scalar_args = [None] * (5 + len(self._mh_state_dict))
        scalar_args.extend(self._data_struct_manager.get_scalar_arg_dtypes())

        if self._in_burnin:
            nmr_iterations = self._nmr_samples
        else:
            nmr_iterations = self._nmr_samples * (self._sample_intervals + 1)
            for item in [self._samples, self._log_likelihoods, self._log_priors]:
                buffer = cl.Buffer(self._cl_run_context.context,
                                   cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                   hostbuf=item)
                data_buffers.append(buffer)
                readout_items.append([buffer, item])
                scalar_args.append(None)

        kernel_func.set_scalar_arg_dtypes(scalar_args)

        kernel_func(
            self._cl_run_context.queue,
            (int(nmr_problems * workgroup_size),),
            (int(workgroup_size),),
            np.uint64(nmr_iterations),
            np.uint64(self._mh_state.nmr_samples_drawn),
            *data_buffers,
            global_offset=(range_start * workgroup_size,))

        for buffer, host_array in readout_items:
            self._enqueue_readout(buffer, host_array, range_start, range_end)

    def _get_buffers(self, workgroup_size):
        data_buffers = []
        readout_items = []

        current_chain_position_buffer = cl.Buffer(self._cl_run_context.context,
                                                  cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                                  hostbuf=self._current_chain_position)
        data_buffers.append(current_chain_position_buffer)
        readout_items.append([current_chain_position_buffer, self._current_chain_position])

        proposal_buffer = cl.Buffer(self._cl_run_context.context,
                                    cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                    hostbuf=self._proposal_state)
        data_buffers.append(proposal_buffer)
        readout_items.append([proposal_buffer, self._proposal_state])

        mcmc_state_buffers = {}
        for mcmc_state_element in sorted(self._mh_state_dict):
            host_array = self._mh_state_dict[mcmc_state_element]['data']

            buffer = cl.Buffer(self._cl_run_context.context,
                               cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                               hostbuf=host_array)
            mcmc_state_buffers[mcmc_state_element] = buffer

            data_buffers.append(buffer)
            readout_items.append([buffer, host_array])

        data_buffers.append(cl.LocalMemory(workgroup_size * np.dtype('double').itemsize))
        data_buffers.extend(self._data_struct_manager.get_kernel_inputs(self._cl_run_context.context, workgroup_size))

        return data_buffers, readout_items

    def _get_mh_state_dict(self):
        """Get a dictionary with the MH state kernel arrays"""
        state_dict = {
            'rng_state': {'data': self._mh_state.get_rng_state(),
                          'cl_type': 'uint'},
            'proposal_state_sampling_counter': {'data': self._mh_state.get_proposal_state_sampling_counter(),
                                                'cl_type': 'ulong'},
            'proposal_state_acceptance_counter': {'data': self._mh_state.get_proposal_state_acceptance_counter(),
                                                  'cl_type': 'ulong'}
        }

        if self._model.proposal_state_update_uses_variance():
            state_dict.update({
                'online_parameter_variance': {'data': self._mh_state.get_online_parameter_variance(),
                                              'cl_type': 'mot_float_type'},
                'online_parameter_variance_update_m2': {
                    'data': self._mh_state.get_online_parameter_variance_update_m2(), 'cl_type': 'mot_float_type'},
                'online_parameter_mean': {'data': self._mh_state.get_online_parameter_mean(),
                                          'cl_type': 'mot_float_type'},
            })

        return state_dict


class _MCMCKernelBuilder(object):

    def __init__(self, compile_flags, mh_state_dict, cl_environment, model,
                 nmr_samples, store_samples, sample_intervals, use_adaptive_proposals, nmr_params,
                 mot_float_dtype):
        self._cl_run_context = cl_environment.get_cl_context()
        self._compile_flags = compile_flags
        self._mh_state_dict = mh_state_dict
        self._cl_environment = cl_environment
        self._model = model
        self._data_info = self._model.get_kernel_data()
        self._data_struct_manager = KernelInputDataManager(self._data_info, mot_float_dtype)
        self._nmr_params = nmr_params
        self._nmr_samples = nmr_samples
        self._store_samples = store_samples
        self._sample_intervals = sample_intervals
        self._use_adaptive_proposals = use_adaptive_proposals
        self._update_parameter_variances = self._model.proposal_state_update_uses_variance()
        self._prior_func = self._model.get_log_prior_function(address_space_parameter_vector='local')
        self._proposal_func = self._model.get_proposal_function(address_space_proposal_state='global')
        self._proposal_logpdf_func = self._model.get_proposal_logpdf(address_space_proposal_state='global')
        self._proposal_state_update_func = self._model.get_proposal_state_update_function(address_space='global')

    def build(self):
        """Build the kernel according to the desired specifications.

        Returns:
            cl.Program: a compiled CL kernel
        """
        return self._compile_kernel(self._get_kernel_source())

    def _compile_kernel(self, kernel_source):
        from mot import configuration
        if configuration.should_ignore_kernel_compile_warnings():
            warnings.simplefilter("ignore")
        return cl.Program(self._cl_run_context.context, kernel_source).build(' '.join(self._compile_flags))

    def _get_kernel_source(self):

        kernel_param_names = [
            'ulong nmr_iterations',
            'ulong iteration_offset']

        kernel_param_names.extend([
            'global mot_float_type* restrict current_chain_position',
            'global mot_float_type* restrict global_proposal_state'])

        for mcmc_state_element in sorted(self._mh_state_dict):
            cl_type = self._mh_state_dict[mcmc_state_element]['cl_type']
            kernel_param_names.append('global {}* restrict global_{}'.format(cl_type, mcmc_state_element))

        kernel_param_names.append('local double* restrict log_likelihood_tmp')
        kernel_param_names.extend(self._data_struct_manager.get_kernel_arguments())

        if self._store_samples:
            kernel_param_names.append('global mot_float_type* restrict samples')
            kernel_param_names.append('global mot_float_type* restrict log_likelihoods')
            kernel_param_names.append('global mot_float_type* restrict log_priors')

        proposal_state_size = self._model.get_proposal_state().shape[1]

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + str(self._model.get_nmr_inst_per_problem()) + '''
        '''
        kernel_source += get_float_type_def(self._model.double_precision)
        kernel_source += self._get_rng_functions()

        kernel_source += self._data_struct_manager.get_struct_definition()
        kernel_source += self._prior_func.get_cl_code()
        kernel_source += self._proposal_func.get_cl_code()

        if self._use_adaptive_proposals:
            kernel_source += self._proposal_state_update_func.get_cl_code()

        if not self._model.is_proposal_symmetric():
            kernel_source += self._proposal_logpdf_func.get_cl_code()

        kernel_source += self._get_log_likelihood_functions()
        kernel_source += self._get_state_update_functions()

        if self._update_parameter_variances:
            kernel_source += self._chain_statistics_update_function()

        kernel_source += '''
            void _sample(local mot_float_type* const x_local,
                         void* rng_data,
                         local double* const current_likelihood,
                         local mot_float_type* const current_prior,
                         mot_data_struct* data,
                         ulong nmr_iterations,
                         ulong iteration_offset,
                         global mot_float_type* const proposal_state,
                         global ulong* const sampling_counter,
                         global ulong* const acceptance_counter,
                         ''' + ('''global mot_float_type* parameter_mean,
                                   global mot_float_type* parameter_variance,
                                   global mot_float_type* parameter_variance_update_m2,'''
                                if self._update_parameter_variances else '') + '''
                         ''' + ('global mot_float_type* samples, '
                                'global mot_float_type* log_likelihoods, '
                                'global mot_float_type* log_priors, '
                                if self._store_samples else '') + '''
                         local double* log_likelihood_tmp){

                ulong i;
                uint j;
                ulong problem_ind = (ulong)(get_global_id(0) / get_local_size(0));
                bool is_first_work_item = get_local_id(0) == 0;

                for(i = 0; i < nmr_iterations; i++){
                '''
        if self._store_samples:
            kernel_source += '''
                    if(is_first_work_item){
                        if(i % ''' + str(self._sample_intervals + 1) + ''' == 0){

                            log_likelihoods[problem_ind * ''' + str(self._nmr_samples) + '''
                                            + (ulong)(i / ''' + str(self._sample_intervals + 1) + ''')
                                ] = *current_likelihood;
                            log_priors[problem_ind * ''' + str(self._nmr_samples) + '''
                                       + (ulong)(i / ''' + str(self._sample_intervals + 1) + ''')
                                ] = *current_prior;

                            for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                                samples[(ulong)(i / ''' + str(self._sample_intervals + 1) + ''') // remove the interval
                                        + j * ''' + str(self._nmr_samples) + '''  // parameter index
                                        + problem_ind * ''' + str(self._nmr_params * self._nmr_samples) + '''
                                    ] = x_local[j];
                            }
                        }
                    }
            '''
        if self._update_parameter_variances:
            kernel_source += '''
                    if(is_first_work_item){
                        for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                            _update_chain_statistics(i + iteration_offset,
                                x_local[j], parameter_mean + j, parameter_variance + j,
                                parameter_variance_update_m2 + j);
                        }
                    }
        '''
        kernel_source += '''
                    _update_state(x_local, rng_data, current_likelihood, current_prior,
                                  data, proposal_state, acceptance_counter,
                                  log_likelihood_tmp);

                    if(is_first_work_item){
                        for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                            sampling_counter[j]++;
                        }

                        ''' + (self._proposal_state_update_func.get_cl_function_name() +
                               '(proposal_state, sampling_counter, acceptance_counter' +
                                (', parameter_variance' if self._update_parameter_variances else '')
                                    + ');'
                               if self._use_adaptive_proposals else '') + '''
                    }
                }
            }
        '''
        kernel_source += '''
            __kernel void sample(
                ''' + ",\n".join(kernel_param_names) + '''
                ){

                ulong problem_ind = (ulong)(get_global_id(0) / get_local_size(0));

                mot_data_struct data = ''' + self._data_struct_manager.get_struct_init_string('problem_ind') + ''';

                rand123_data rand123_rng_data = _rng_data_from_array(global_rng_state);
                void* rng_data = (void*)&rand123_rng_data;

                local mot_float_type x_local[''' + str(self._nmr_params) + '''];

                local double current_likelihood;
                local mot_float_type current_prior;

                global mot_float_type* proposal_state =
                    global_proposal_state + problem_ind * ''' + str(proposal_state_size) + ''';
                global ulong* sampling_counter =
                    global_proposal_state_sampling_counter + problem_ind * ''' + str(self._nmr_params) + ''';
                global ulong* acceptance_counter =
                    global_proposal_state_acceptance_counter + problem_ind * ''' + str(self._nmr_params) + ''';
                '''
        if self._update_parameter_variances:
            kernel_source += '''
                global mot_float_type* parameter_mean =
                    global_online_parameter_mean + problem_ind * ''' + str(self._nmr_params) + ''';
                global mot_float_type* parameter_variance =
                    global_online_parameter_variance + problem_ind * ''' + str(self._nmr_params) + ''';
                global mot_float_type* parameter_variance_update_m2 =
                    global_online_parameter_variance_update_m2 + problem_ind * ''' + str(self._nmr_params) + ''';
            '''
        kernel_source += '''

                if(get_local_id(0) == 0){
                    for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x_local[i] = current_chain_position[problem_ind * ''' + str(self._nmr_params) + ''' + i];
                    }

                    current_prior = ''' + self._prior_func.get_cl_function_name() + '''(&data, x_local);
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                _fill_log_likelihood_tmp(&data, x_local, log_likelihood_tmp);
                _sum_log_likelihood_tmp(log_likelihood_tmp, &current_likelihood);

                _sample(x_local, rng_data, &current_likelihood, &current_prior, &data, nmr_iterations,
                        iteration_offset, proposal_state, sampling_counter, acceptance_counter,
                    ''' + ('parameter_mean, parameter_variance, parameter_variance_update_m2,'
                            if self._update_parameter_variances else '') + '''
                    ''' + ('samples, log_likelihoods, log_priors, ' if self._store_samples else '') + '''
                    log_likelihood_tmp);

                if(get_local_id(0) == 0){
                    for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        current_chain_position[problem_ind * ''' + str(self._nmr_params) + ''' + i] = x_local[i];
                    }

                    _rng_data_to_array(rand123_rng_data, global_rng_state);
                }
            }
        '''
        return kernel_source

    def _get_rng_functions(self):
        random_library = Rand123()
        kernel_source = random_library.get_cl_code()

        kernel_source += '''
            rand123_data _rng_data_from_array(global uint* rng_state){
                ulong problem_ind = (ulong)(get_global_id(0) / get_local_size(0));

                return rand123_initialize_data(
                    (uint[]){rng_state[0 + problem_ind * 6],
                             rng_state[1 + problem_ind * 6],
                             rng_state[2 + problem_ind * 6],
                             rng_state[3 + problem_ind * 6],
                             rng_state[4 + problem_ind * 6],
                             rng_state[5 + problem_ind * 6],
                             0, 0}); // last two states reserved for future use
            }

            void _rng_data_to_array(rand123_data data, global uint* rng_state){
                ulong problem_ind = (ulong)(get_global_id(0) / get_local_size(0));

                uint state[8]; // only using 6, other two reserved for future use
                rand123_data_to_array(data, state);

                for(uint i = 0; i < 6; i++){
                    rng_state[i + problem_ind * 6] = state[i];
                }
            }
        '''
        return kernel_source

    def _get_log_likelihood_functions(self):
        ll_func = self._model.get_log_likelihood_per_observation_function()

        kernel_source = ll_func.get_cl_code()
        kernel_source += '''
            void _fill_log_likelihood_tmp(mot_data_struct* data,
                                          local mot_float_type* const x_local,
                                          local double* log_likelihood_tmp){

                ulong observation_ind;
                ulong local_id = get_local_id(0);
                log_likelihood_tmp[local_id] = 0;
                uint workgroup_size = get_local_size(0);
                uint elements_for_workitem = ceil(NMR_INST_PER_PROBLEM / (mot_float_type)workgroup_size);

                if(workgroup_size * (elements_for_workitem - 1) + local_id >= NMR_INST_PER_PROBLEM){
                    elements_for_workitem -= 1;
                }

                mot_float_type x_private[''' + str(self._nmr_params) + '''];
                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    x_private[i] = x_local[i];
                }

                for(uint i = 0; i < elements_for_workitem; i++){
                    observation_ind = i * workgroup_size + local_id;

                    log_likelihood_tmp[local_id] += ''' + ll_func.get_cl_function_name() + '''(
                        data, x_private, observation_ind);
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            void _sum_log_likelihood_tmp(local double* log_likelihood_tmp, local double* log_likelihood){
                *log_likelihood = 0;
                for(uint i = 0; i < get_local_size(0); i++){
                    *log_likelihood += log_likelihood_tmp[i];
                }
            }
        '''
        return kernel_source

    def _get_state_update_functions(self):
        kernel_source = '''
            void _update_state(local mot_float_type* const x_local,
                               void* rng_data,
                               local double* const current_likelihood,
                               local mot_float_type* const current_prior,
                               mot_data_struct* data,
                               global mot_float_type* const proposal_state,
                               global ulong * const acceptance_counter,
                               local double* log_likelihood_tmp){

                local float4 random_nmr;
                local mot_float_type new_prior;
                local double new_likelihood;
                local double bayesian_f;
                mot_float_type old_x;
                bool is_first_work_item = get_local_id(0) == 0;

                #pragma unroll 1
                for(uint k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                    if(is_first_work_item){
                        random_nmr = frand(rng_data);

                        old_x = x_local[k];
                        x_local[k] = ''' + self._proposal_func.get_cl_function_name() + \
                            '''(k, x_local[k], rng_data, proposal_state);

                        new_prior = ''' + self._prior_func.get_cl_function_name() + '''(data, x_local);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    if(exp(new_prior) > 0){
                        _fill_log_likelihood_tmp(data, x_local, log_likelihood_tmp);

                        if(is_first_work_item){
                            _sum_log_likelihood_tmp(log_likelihood_tmp, &new_likelihood);

        '''
        if self._model.is_proposal_symmetric():
            kernel_source += '''
                            bayesian_f = exp((new_likelihood + new_prior) - (*current_likelihood + *current_prior));
                '''
        else:
            kernel_source += '''
                            mot_float_type x_to_prop = ''' + \
                             self._proposal_logpdf_func.get_cl_function_name() + '''(
                                k, old_x, x_local[k], proposal_state);
                            mot_float_type prop_to_x = ''' + \
                             self._proposal_logpdf_func.get_cl_function_name() + '''(
                                k, x_local[k], x_local[k], proposal_state);

                            bayesian_f = exp((new_likelihood + new_prior + x_to_prop) -
                                                (*current_likelihood + *current_prior + prop_to_x));
                '''
        kernel_source += '''
                            if(random_nmr.x < bayesian_f){
                                *current_likelihood = new_likelihood;
                                *current_prior = new_prior;
                                acceptance_counter[k]++;
                            }
                            else{
                                x_local[k] = old_x;
                            }
                        }
                    }
                    else{ // prior returned 0
                        if(is_first_work_item){
                            x_local[k] = old_x;
                        }
                    }
                }
            }
        '''
        return kernel_source

    def _chain_statistics_update_function(self):
        kernel_source = '''
            /** Online variance algorithm by Welford
             *  B. P. Welford (1962)."Note on a method for calculating corrected sums of squares
             *      and products". Technometrics 4(3):419-420.
             *
             * Also studied in:
             * Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1983).
             *      Algorithms for Computing the Sample Variance: Analysis and Recommendations.
             *      The American Statistician 37, 242-247. http://www.jstor.org/stable/2683386
             */
            void _update_chain_statistics(const ulong chain_count,
                                          const mot_float_type new_param_value,
                                          global mot_float_type* const parameter_mean,
                                          global mot_float_type* const parameter_variance,
                                          global mot_float_type* const parameter_variance_update_m2){

                mot_float_type previous_mean = *parameter_mean;
                *parameter_mean += (new_param_value - *parameter_mean) / (chain_count + 1);
                *parameter_variance_update_m2 += (new_param_value - previous_mean)
                                                    * (new_param_value - *parameter_mean);

                if(chain_count > 1){
                    *parameter_variance = *parameter_variance_update_m2 / (chain_count - 1);
                }
            }
        '''
        return kernel_source


class MHState(object):
    """The Metropolis Hastings state is used to initialize the state of the MH sampler.

    The state is stored at the end of every MH run and can be used to continue sampling again from the
    previous end point.
    """

    @property
    def nmr_samples_drawn(self):
        """Get the amount of samples already drawn, i.e. at what point in time is this state.

        Returns:
            uint64: the current number of samples already drawn before this state
        """
        raise NotImplementedError()

    def get_proposal_state_sampling_counter(self):
        """Get the current state of the sampling counter that can be used by the adaptive proposals.

        This value is per problem instance passed to the adaptive proposals which may reset the value.

        Returns:
            ndarray: a (d, p) array with for d problems and p parameters the current sampling counter,
                should be of a np.uint64 type.
        """
        raise NotImplementedError()

    def get_proposal_state_acceptance_counter(self):
        """Get the current state of the acceptance counter that can be used by the adaptive proposals.

        This value is per problem instance passed to the adaptive proposals which may reset the value.

        Returns:
            ndarray: a (d, p) array with for d problems and p parameters the current acceptance counter,
                should be of a np.uint64 type.
        """
        raise NotImplementedError()

    def get_online_parameter_variance(self):
        """Get the current state of the online parameter variance that can be used by the adaptive proposals.

        This value is updated while sampling and is passed as a constant to the adaptive proposals.

        Returns:
            ndarray: a (d, p) array with for d problems and p parameters the current parameter variance,
                should be of a np.float32 or np.float64 type (it will still be auto-converted to the
                current double type in the MCMC function).
        """
        raise NotImplementedError()

    def get_online_parameter_variance_update_m2(self):
        """A helper variable used in updating the online parameter variance.

        Returns:
            ndarray: a (d, p) array with for d problems and p parameters the current M2 state,
                should be of a np.float32 or np.float64 type (it will still be auto-converted to the
                current double type in the MCMC function).
        """
        raise NotImplementedError()

    def get_online_parameter_mean(self):
        """Get the current state of the online parameter mean, a helper variance in updating the variance.

        Returns:
            ndarray: a (d, p) array with for d problems and p parameters the current parameter mean,
                should be of a np.float32 or np.float64 type (it will still be auto-converted to the
                current double type in the MCMC function).
        """
        raise NotImplementedError()

    def get_rng_state(self):
        """Get the RNG state array for every problem instance.

        Returns:
            ndarray: a (d, \*) state array with for every d problem the state of size > 0
        """
        raise NotImplementedError()


class DefaultMHState(MHState):

    def __init__(self, nmr_problems, nmr_params, double_precision=False):
        """Creates a initial (default) MCMC state.

        Args:
            nmr_problems (int): the number of problems we are optimizing, used to create the default state.
            nmr_params (int): the number of parameters in the model, used to create the default state.
            double_precision (boolean): used when auto-creating some of the default state items.
        """
        self._nmr_problems = nmr_problems
        self._nmr_params = nmr_params
        self._double_precision = double_precision

        self._float_dtype = np.float32
        if double_precision:
            self._float_dtype = np.float64

    @property
    def nmr_samples_drawn(self):
        return 0

    def get_proposal_state_sampling_counter(self):
        return np.zeros((self._nmr_problems, self._nmr_params), dtype=np.uint64, order='C')

    def get_proposal_state_acceptance_counter(self):
        return np.zeros((self._nmr_problems, self._nmr_params), dtype=np.uint64, order='C')

    def get_online_parameter_variance(self):
        return np.zeros((self._nmr_problems, self._nmr_params), dtype=self._float_dtype, order='C')

    def get_online_parameter_variance_update_m2(self):
        return np.zeros((self._nmr_problems, self._nmr_params), dtype=self._float_dtype, order='C')

    def get_online_parameter_mean(self):
        return np.zeros((self._nmr_problems, self._nmr_params), dtype=self._float_dtype, order='C')

    def get_rng_state(self):
        dtype_info = np.iinfo(np.uint32)
        return np.random.uniform(low=dtype_info.min, high=dtype_info.max + 1,
                                 size=(self._nmr_problems, 6)).astype(np.uint32)


class SimpleMHState(MHState):
    def __init__(self, nmr_samples_drawn, proposal_state_sampling_counter, proposal_state_acceptance_counter,
                 online_parameter_variance, online_parameter_variance_update_m2,
                 online_parameter_mean,
                 rng_state):
        """A simple MCMC state containing provided items

        Args:
            nmr_samples_drawn (ndarray): the current number of samples already drawn to reach this state.
            proposal_state_sampling_counter (ndarray): a (d, p) array with for d problems and p parameters
                the current sampling counter.
            proposal_state_acceptance_counter (ndarray): a (d, p) array with for d problems and p parameters
                the current acceptance counter.
            online_parameter_variance (ndarray): a (d, p) array with for d problems and p parameters
                the current state of the online parameter variance
            online_parameter_variance_update_m2 (ndarray): a (d, p) array with for d problems and p parameters
                the current state of the online parameter variance update variable.
            online_parameter_mean (ndarray): a (d, p) array with for d problems and p parameters
                the current state of the online parameter mean
            rng_state (ndarray): a (d, \*) array with for d problems the rng state vector
        """
        self._nmr_samples_drawn = nmr_samples_drawn
        self._proposal_state_sampling_counter = proposal_state_sampling_counter
        self._proposal_state_acceptance_counter = proposal_state_acceptance_counter
        self._online_parameter_variance = online_parameter_variance
        self._online_parameter_variance_update_m2 = online_parameter_variance_update_m2
        self._online_parameter_mean = online_parameter_mean
        self._rng_state = rng_state

    @property
    def nmr_samples_drawn(self):
        return self._nmr_samples_drawn

    def with_nmr_samples_drawn(self, nmr_samples_drawn):
        """Recreate this object and set the number of samples drawn to the specified value."""
        return type(self)(
            nmr_samples_drawn,
            self.get_proposal_state_sampling_counter(),
            self.get_proposal_state_acceptance_counter(),
            self.get_online_parameter_variance(),
            self.get_online_parameter_variance_update_m2(),
            self.get_online_parameter_mean(),
            self.get_rng_state()
        )

    def get_proposal_state_sampling_counter(self):
        return self._proposal_state_sampling_counter

    def get_proposal_state_acceptance_counter(self):
        return self._proposal_state_acceptance_counter

    def get_online_parameter_variance(self):
        return self._online_parameter_variance

    def get_online_parameter_variance_update_m2(self):
        return self._online_parameter_variance_update_m2

    def get_online_parameter_mean(self):
        return self._online_parameter_mean

    def get_rng_state(self):
        return self._rng_state


def _prepare_mh_state(mh_state, float_dtype):
    """Return a new MH state in which all the state variables are sanitized to the correct data type.

    Args:
        mh_state (MHState): the MH state we wish to sanitize
        float_dtype (dtype): the numpy dtype for the floats, either np.float32 or np.float64

    Returns:
        SimpleMHState: MH state with the same data only then possibly sanitized
    """
    proposal_state_sampling_counter = np.require(np.copy(mh_state.get_proposal_state_sampling_counter()),
                                                 np.uint64,requirements=['C', 'A', 'O', 'W'])

    proposal_state_acceptance_counter = np.require(np.copy(mh_state.get_proposal_state_acceptance_counter()),
                                                   np.uint64, requirements=['C', 'A', 'O', 'W'])

    online_parameter_variance = np.require(np.copy(mh_state.get_online_parameter_variance()),
                                           float_dtype, requirements=['C', 'A', 'O', 'W'])

    online_parameter_variance_update_m2 = np.require(np.copy(mh_state.get_online_parameter_variance_update_m2()),
                                                     float_dtype, requirements=['C', 'A', 'O', 'W'])

    online_parameter_mean = np.require(np.copy(mh_state.get_online_parameter_mean()), float_dtype,
                                       requirements=['C', 'A', 'O', 'W'])

    rng_state = np.require(np.copy(mh_state.get_rng_state()), np.uint32, requirements=['C', 'A', 'O', 'W'])

    return SimpleMHState(mh_state.nmr_samples_drawn,
                         proposal_state_sampling_counter, proposal_state_acceptance_counter,
                         online_parameter_variance, online_parameter_variance_update_m2,
                         online_parameter_mean,
                         rng_state
                         )

