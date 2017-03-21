import gc
import pyopencl as cl
import numpy as np
from mot.cl_routines.mapping.error_measures import ErrorMeasures
from mot.cl_routines.mapping.residual_calculator import ResidualCalculator
from mot.random123 import get_random123_cl_code, RandomStartingPoint
from mot.mcmc_diagnostics import multivariate_ess, univariate_ess
from ...utils import results_to_dict, get_float_type_def
from ...load_balance_strategies import Worker
from ...cl_routines.sampling.base import AbstractSampler


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MetropolisHastings(AbstractSampler):

    def __init__(self, cl_environments=None, load_balancer=None, nmr_samples=None, burn_length=None,
                 sample_intervals=None, use_adaptive_proposals=True, **kwargs):
        """An CL implementation of Metropolis Hastings.

        Args:
            cl_environments: a list with the cl environments to use
            load_balancer: the load balance strategy to use
            nmr_samples (int): The length of the (returned) chain per voxel, defaults to 0
            burn_length (int): The length of the burn in (per voxel), these are extra samples,
                jump is set to 1 (no thinning)
            sample_intervals (int): how many sample we wait before storing one.
                This will draw extra samples (chain_length * sample_intervals). If set to zero we
                store every sample after the burn in.
            use_adaptive_proposals (boolean): if we use the adaptive proposals (set to True) or not (set to False).

        Attributes:
            nmr_samples (int): The length of the (returned) chain per voxel
            burn_length (int): The length of the burn in (per voxel), these are extra samples,
                jump is set to 1 (no thinning)
            sample_intervals (int): how many samples we wait before we take,
                this will draw extra samples (chain_length * sample_intervals)
            proposal_update_intervals (int): after how many samples we would like to update the proposals
                This is during burning and sampling. A value of 1 means update after every jump.
        """
        super(MetropolisHastings, self).__init__(cl_environments=cl_environments, load_balancer=load_balancer, **kwargs)
        self._nmr_samples = nmr_samples or 500
        self.burn_length = burn_length
        self.sample_intervals = sample_intervals
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

    @nmr_samples.setter
    def nmr_samples(self, value):
        self._nmr_samples = value or 1

    def sample(self, model, init_params=None, full_output=False):
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        self._do_initial_logging(model)

        current_chain_position = np.require(model.get_initial_parameters(init_params), np_dtype,
                                            requirements=['C', 'A', 'O', 'W'])
        samples = np.zeros((model.get_nmr_problems(), current_chain_position.shape[1], self.nmr_samples),
                           dtype=np_dtype, order='C')

        proposal_state = np.require(model.get_proposal_state(), np_dtype, requirements=['C', 'A', 'O', 'W'])
        mcmc_state = self._get_mcmc_state(model, current_chain_position)

        self._logger.info('Starting sampling with method {0}'.format(self.__class__.__name__))

        workers = self._create_workers(lambda cl_environment: _MHWorker(
            cl_environment, self.get_compile_flags_list(model.double_precision), model, current_chain_position,
            samples, proposal_state, mcmc_state,
            self.nmr_samples, self.burn_length, self.sample_intervals, self.use_adaptive_proposals))
        self.load_balancer.process(workers, model.get_nmr_problems())

        del workers
        gc.collect()

        samples_dict = results_to_dict(samples, model.get_optimized_param_names())

        self._logger.info('Finished sampling')

        if full_output:
            self._logger.info('Starting post-sampling transformations')
            volume_maps = model.finalize_optimization_results(model.samples_to_statistics(samples_dict))
            self._logger.info('Finished post-sampling transformations')

            self._logger.info('Calculating errors measures')
            errors = ResidualCalculator(cl_environments=self.cl_environments,
                                        load_balancer=self.load_balancer).calculate(model, volume_maps)
            error_measures = ErrorMeasures(self.cl_environments, self.load_balancer,
                                           model.double_precision).calculate(errors)
            volume_maps.update(error_measures)
            self._logger.info('Done calculating errors measures')

            self._logger.info('Calculating the multivariate ESS')
            mv_ess = multivariate_ess(samples)
            volume_maps.update(MultivariateESS=mv_ess)
            self._logger.info('Finished calculating the multivariate ESS')

            self._logger.info('Calculating the univariate ESS with method \'standard_error\'')
            uv_ess = univariate_ess(samples, method='standard_error')
            uv_ess_maps = results_to_dict(uv_ess, [a + '.UnivariateESS' for a in model.get_optimized_param_names()])
            volume_maps.update(uv_ess_maps)
            self._logger.info('Finished calculating the univariate ESS')

            # todo remove and simplify the MCMC state
            proposal_state_dict = results_to_dict(proposal_state, model.get_proposal_state_names())
            # proposal_state_dict.update(
            #     results_to_dict(proposal_state_sampling_counter, [a + '.sampling_counter' for a in model.get_optimized_param_names()]))
            # proposal_state_dict.update(
            #     results_to_dict(proposal_state_acceptance_counter, [a + '.acceptance_counter' for a in model.get_optimized_param_names()]))
            # proposal_state_dict.update(
            #     results_to_dict(online_parameter_variance,
            #                     [a + '.parameter_variance' for a in model.get_optimized_param_names()]))

            return samples_dict, volume_maps, proposal_state_dict

        return samples_dict

    def _get_mcmc_state(self, model, parameters):
        nmr_params = model.get_nmr_estimable_parameters()
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        proposal_state_sampling_counter = np.zeros((model.get_nmr_problems(), nmr_params),
                                                   dtype=np.uint32, order='C')
        proposal_state_acceptance_counter = np.zeros((model.get_nmr_problems(), nmr_params),
                                                     dtype=np.uint32, order='C')

        state_dict = {
            'proposal_state_sampling_counter': {'data': proposal_state_sampling_counter,
                                                'cl_type': 'uint'},
            'proposal_state_acceptance_counter': {'data': proposal_state_acceptance_counter,
                                                  'cl_type': 'uint'}
        }

        if model.proposal_state_update_uses_variance():
            online_parameter_variance = np.zeros((model.get_nmr_problems(), nmr_params),
                                                 dtype=np_dtype, order='C')
            online_parameter_variance_update_m2 = np.zeros((model.get_nmr_problems(), nmr_params),
                                                           dtype=np_dtype, order='C')
            online_parameter_mean = np.array(parameters, copy=True)

            state_dict.update({
                'online_parameter_variance': {'data': online_parameter_variance, 'cl_type': 'mot_float_type'},
                'online_parameter_variance_update_m2': {'data': online_parameter_variance_update_m2,
                                                        'cl_type': 'mot_float_type'},
                'online_parameter_mean': {'data': online_parameter_mean,
                                          'cl_type': 'mot_float_type'},
            })

        return state_dict

    def _do_initial_logging(self, model):
        self._logger.info('Entered sampling routine.')
        self._logger.info('We will use a {} precision float type for the calculations.'.format(
            'double' if model.double_precision else 'single'))

        for env in self.load_balancer.get_used_cl_environments(self.cl_environments):
            self._logger.info('Using device \'{}\'.'.format(str(env)))

        self._logger.info('Using compile flags: {}'.format(self.get_compile_flags_list(model.double_precision)))

        self._logger.info('The parameters we will sample are: {0}'.format(model.get_optimized_param_names()))

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


class _MHWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, current_chain_position, samples, proposal_state,
                 mcmc_state, nmr_samples, burn_length, sample_intervals, use_adaptive_proposals):
        super(_MHWorker, self).__init__(cl_environment)

        self._model = model
        self._current_chain_position = current_chain_position
        self._nmr_params = current_chain_position.shape[1]
        self._samples = samples
        self._proposal_state = proposal_state
        self._update_parameter_variances = self._model.proposal_state_update_uses_variance()
        self._mcmc_state = mcmc_state

        self._mcmc_state_kernel_order = ['proposal_state_sampling_counter',
                                         'proposal_state_acceptance_counter']
        if self._update_parameter_variances:
            self._mcmc_state_kernel_order.extend(
                ['online_parameter_variance', 'online_parameter_variance_update_m2', 'online_parameter_mean'])

        self._nmr_samples = nmr_samples
        self._burn_length = burn_length
        self._sample_intervals = sample_intervals
        self.use_adaptive_proposals = use_adaptive_proposals

        self._rand123_starting_point = RandomStartingPoint()

        self._kernel = self._build_kernel(compile_flags)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        workgroup_size = cl.Kernel(self._kernel, 'sample').get_work_group_info(
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, self._cl_environment.device)

        data_buffers, samples_buf, proposal_buffer, mcmc_state_buffers, current_chain_position_buffer = \
            self._get_buffers(range_start, range_end, workgroup_size)

        kernel_event = self._kernel.sample(self._cl_run_context.queue,
                                           (int(nmr_problems * workgroup_size),),
                                           (int(workgroup_size),),
                                           *data_buffers)

        return_events = [
            self._enqueue_readout(samples_buf, self._samples, 0, nmr_problems, [kernel_event]),
            self._enqueue_readout(proposal_buffer, self._proposal_state, 0, nmr_problems, [kernel_event]),
            self._enqueue_readout(current_chain_position_buffer, self._current_chain_position, 0, nmr_problems,
                                  [kernel_event]),
        ]

        for mcmc_state_element in self._mcmc_state_kernel_order:
            buffer = mcmc_state_buffers[mcmc_state_element]
            host_array = self._mcmc_state[mcmc_state_element]['data']
            return_events.append(self._enqueue_readout(buffer, host_array, 0, nmr_problems, [kernel_event]))

        return return_events

    def _get_buffers(self, range_start, range_end, workgroup_size):
        data_buffers = []

        current_chain_position_buffer = cl.Buffer(self._cl_run_context.context,
                                                  cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                                  hostbuf=self._current_chain_position[range_start:range_end, :])
        data_buffers.append(current_chain_position_buffer)

        samples_buf = cl.Buffer(self._cl_run_context.context,
                                cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                hostbuf=self._samples[range_start:range_end, ...])
        data_buffers.append(samples_buf)

        proposal_buffer = cl.Buffer(self._cl_run_context.context,
                                    cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                    hostbuf=self._proposal_state[range_start:range_end, ...])
        data_buffers.append(proposal_buffer)

        mcmc_state_buffers = {}
        for mcmc_state_element in self._mcmc_state_kernel_order:
            host_array = self._mcmc_state[mcmc_state_element]['data']

            buffer = cl.Buffer(self._cl_run_context.context,
                               cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                               hostbuf=host_array[range_start:range_end, ...])
            mcmc_state_buffers[mcmc_state_element] = buffer
            data_buffers.append(buffer)

        data_buffers.append(cl.LocalMemory(workgroup_size * np.dtype('double').itemsize))

        for data in self._model.get_data():
            data_buffers.append(cl.Buffer(self._cl_run_context.context,
                                          cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data))

        return data_buffers, samples_buf, proposal_buffer, mcmc_state_buffers, current_chain_position_buffer

    def _get_kernel_source(self):
        kernel_param_names = ['global mot_float_type* current_chain_position',
                              'global mot_float_type* samples',
                              'global mot_float_type* global_proposal_state']

        for mcmc_state_element in self._mcmc_state_kernel_order:
            cl_type = self._mcmc_state[mcmc_state_element]['cl_type']
            kernel_param_names.append('global {}* global_{}'.format(cl_type, mcmc_state_element))

        kernel_param_names.append('local double* log_likelihood_tmp')

        kernel_param_names.extend(self._model.get_kernel_param_names(self._cl_environment.device))

        proposal_state_size = self._model.get_proposal_state().shape[1]

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + str(self._model.get_nmr_inst_per_problem()) + '''
        '''
        kernel_source += get_random123_cl_code()
        kernel_source += get_float_type_def(self._model.double_precision)
        kernel_source += self._model.get_kernel_data_struct(self._cl_environment.device)
        kernel_source += self._model.get_log_prior_function('getLogPrior', address_space_parameter_vector='local')
        kernel_source += self._model.get_proposal_function('getProposal', address_space_proposal_state='global')
        kernel_source += self._model.get_log_likelihood_per_observation_function(
            'getLogLikelihoodPerObservation', full_likelihood=False)

        if self.use_adaptive_proposals:
            kernel_source += self._model.get_proposal_state_update_function('updateProposalState',
                                                                            address_space='global')

        if not self._model.is_proposal_symmetric():
            kernel_source += self._model.get_proposal_logpdf('getProposalLogPDF', address_space_proposal_state='global')

        kernel_source += '''
            void _fill_log_likelihood_tmp(const void* const data,
                                          local mot_float_type* const x_local,
                                          local double* log_likelihood_tmp){

                int observation_ind;
                uint local_id = get_local_id(0);
                log_likelihood_tmp[local_id] = 0;
                uint workgroup_size = get_local_size(0);

                mot_float_type x_private[''' + str(self._nmr_params) + '''];
                for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    x_private[i] = x_local[i];
                }

                for(int i = 0; i < ceil(NMR_INST_PER_PROBLEM /
                                        (mot_float_type)workgroup_size); i++){

                    observation_ind = i * workgroup_size + local_id;

                    if(observation_ind < NMR_INST_PER_PROBLEM){
                        log_likelihood_tmp[local_id] += getLogLikelihoodPerObservation(
                            data, x_private, observation_ind);
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            void _sum_log_likelihood_tmp_local(local double* log_likelihood_tmp, local double* log_likelihood){
                *log_likelihood = 0;
                for(int i = 0; i < get_local_size(0); i++){
                    *log_likelihood += log_likelihood_tmp[i];
                }
            }

            void _sum_log_likelihood_tmp_private(local double* log_likelihood_tmp, private double* log_likelihood){
                *log_likelihood = 0;
                for(int i = 0; i < get_local_size(0); i++){
                    *log_likelihood += log_likelihood_tmp[i];
                }
            }
        '''

        kernel_source += '''
            void _update_state(local mot_float_type* const x_local,
                               void* rng_data,
                               local double* const current_likelihood,
                               local mot_float_type* const current_prior,
                               const void* const data,
                               global mot_float_type* const proposal_state,
                               global uint * const acceptance_counter,
                               local double* log_likelihood_tmp){

                float4 random_nmr;
                mot_float_type new_prior = 0;
                double new_likelihood;
                double bayesian_f;
                mot_float_type old_x;
                bool is_first_work_item = get_local_id(0) == 0;

                #pragma unroll 1
                for(int k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                    if(is_first_work_item){
                        random_nmr = frand(rng_data);

                        old_x = x_local[k];
                        x_local[k] = getProposal(k, x_local[k], rng_data, proposal_state);

                        new_prior = getLogPrior(data, x_local);
                    }

                    if(exp(new_prior) > 0){
                        _fill_log_likelihood_tmp(data, x_local, log_likelihood_tmp);

                        if(is_first_work_item){
                            _sum_log_likelihood_tmp_private(log_likelihood_tmp, &new_likelihood);

        '''
        if self._model.is_proposal_symmetric():
            kernel_source += '''
                            bayesian_f = exp((new_likelihood + new_prior) - (*current_likelihood + *current_prior));
                '''
        else:
            kernel_source += '''
                            mot_float_type x_to_prop = getProposalLogPDF(k, old_x, x_local[k], proposal_state);
                            mot_float_type prop_to_x = getProposalLogPDF(k, x_local[k], x_local[k], proposal_state);

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
        kernel_source += '''
            void _sample(local mot_float_type* const x_local,
                         void* rng_data,
                         local double* const current_likelihood,
                         local mot_float_type* const current_prior,
                         const void* const data,
                         global mot_float_type* const proposal_state,
                         global uint* const sampling_counter,
                         global uint* const acceptance_counter,
                         ''' + ('''global mot_float_type* parameter_mean,
                                   global mot_float_type* parameter_variance,
                                   global mot_float_type* parameter_variance_update_m2,'''
                                if self._update_parameter_variances else '') + '''
                         global mot_float_type* samples,
                         local double* log_likelihood_tmp){

                uint i, j;
                uint problem_ind = get_group_id(0);
                bool is_first_work_item = get_local_id(0) == 0;
                mot_float_type previous_mean;

                for(i = 0; i < ''' + str(self._nmr_samples * (self._sample_intervals + 1)
                                         + self._burn_length) + '''; i++){

                    _update_state(x_local, rng_data, current_likelihood, current_prior,
                                  data, proposal_state, acceptance_counter,
                                  log_likelihood_tmp);

                    if(is_first_work_item){
                        for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                            sampling_counter[j]++;
                        }

                        ''' + ('updateProposalState(proposal_state, sampling_counter, acceptance_counter' +
                                (', parameter_variance' if self._update_parameter_variances else '')
                                    + ');'
                               if self.use_adaptive_proposals else '') + '''

                        '''
        if self._update_parameter_variances:
            kernel_source += '''
                        if(i > 0){
                            for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                                // online variance, algorithm by Welford
                                //  B. P. Welford (1962)."Note on a method for calculating corrected sums of squares
                                //      and products". Technometrics 4(3):419–420.
                                //
                                // also studied in:
                                // Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1983).
                                //      Algorithms for Computing the Sample Variance: Analysis and Recommendations.
                                //      The American Statistician 37, 242-247. http://www.jstor.org/stable/2683386

                                previous_mean = parameter_mean[j];
                                parameter_mean[j] += (x_local[j] - parameter_mean[j]) / (i + 1);
                                parameter_variance_update_m2[j] += (x_local[j] - previous_mean)
                                                                      * (x_local[j] - parameter_mean[j]);

                                if(i > 1){
                                    parameter_variance[j] = parameter_variance_update_m2[j] / (i - 1);
                                }
                            }
                        }
            '''
        kernel_source += '''
                        if(i >= ''' + str(self._burn_length) + ''' &&
                            i % ''' + str(self._sample_intervals + 1) + ''' == 0){

                            for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                                samples[(uint)((i - ''' + str(self._burn_length) + ''') // remove the burn-in
                                                / ''' + str(self._sample_intervals + 1) + ''') // remove the interval
                                        + j * ''' + str(self._nmr_samples) + '''  // parameter index
                                        + problem_ind * ''' + str(self._nmr_params * self._nmr_samples) + ''']
                                            = x_local[j];
                            }
                        }
                    }
                }
            }
        '''

        kernel_source += '''
            __kernel void sample(
                ''' + ",\n".join(kernel_param_names) + '''
                ){

                uint problem_ind = get_group_id(0);
                const uint nmr_params = ''' + str(self._nmr_params) + ''';

                ''' + self._model.get_kernel_data_struct_initialization(self._cl_environment.device,
                                                                        'data', 'problem_ind') + '''

                rand123_data rand123_rng_data = ''' + self._get_rand123_init_cl_code() + ''';
                void* rng_data = (void*)&rand123_rng_data;

                local mot_float_type x_local[nmr_params];

                local double current_likelihood;
                local mot_float_type current_prior;

                global mot_float_type* proposal_state =
                    global_proposal_state + problem_ind * ''' + str(proposal_state_size) + ''';
                global mot_float_type* sampling_counter =
                    global_proposal_state_sampling_counter + problem_ind * nmr_params;
                global mot_float_type* acceptance_counter =
                    global_proposal_state_acceptance_counter + problem_ind * nmr_params;
                '''
        if self._update_parameter_variances:
            kernel_source += '''
                global mot_float_type* parameter_mean =
                    global_online_parameter_mean + problem_ind * nmr_params;
                global mot_float_type* parameter_variance =
                    global_online_parameter_variance + problem_ind * nmr_params;
                global mot_float_type* parameter_variance_update_m2 =
                    global_online_parameter_variance_update_m2 + problem_ind * nmr_params;
            '''
        kernel_source += '''

                if(get_local_id(0) == 0){
                    for(int i = 0; i < nmr_params; i++){
                        x_local[i] = current_chain_position[problem_ind * nmr_params + i];
                    }

                    current_prior = getLogPrior((void*)&data, x_local);
                }

                _fill_log_likelihood_tmp((void*)&data, x_local, log_likelihood_tmp);
                _sum_log_likelihood_tmp_local(log_likelihood_tmp, &current_likelihood);

                _sample(x_local, rng_data, &current_likelihood,
                    &current_prior, (void*)&data, proposal_state, sampling_counter, acceptance_counter,
                    ''' + ('parameter_mean, parameter_variance, parameter_variance_update_m2,'
                            if self._update_parameter_variances else '') + '''
                    samples, log_likelihood_tmp);

                if(get_local_id(0) == 0){
                    for(int i = 0; i < nmr_params; i++){
                        current_chain_position[problem_ind * nmr_params + i] = x_local[i];
                    }
                }
            }
        '''
        return kernel_source

    def _get_rand123_init_cl_code(self):
        key = self._rand123_starting_point.get_key()
        counter = self._rand123_starting_point.get_counter()

        if len(key):
            return 'rand123_initialize_data_extra_precision((uint[]){%(c0)r, %(c1)r, %(c2)r, %(c3)r}, ' \
                   '(uint[]){%(k0)r, %(k1)r})' % {'c0': counter[0], 'c1': counter[1],
                                                  'c2': counter[2], 'c3': counter[3],
                                                  'k0': key[0], 'k1': counter[1]}
        else:
            return 'rand123_initialize_data((uint[]){%(c0)r, %(c1)r, %(c2)r, %(c3)r})' \
                   % {'c0': counter[0], 'c1': counter[1], 'c2': counter[2], 'c3': counter[3]}

