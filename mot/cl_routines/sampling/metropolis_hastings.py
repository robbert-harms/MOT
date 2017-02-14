import math
import pyopencl as cl
import numpy as np
from mot.cl_routines.mapping.error_measures import ErrorMeasures
from mot.cl_routines.mapping.residual_calculator import ResidualCalculator
from mot.random123 import get_random123_cl_code, RandomStartingPoint
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
            nmr_samples (int): The length of the (returned) chain per voxel, defaults to 500
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
            self.burn_length = 500

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

        parameters = model.get_initial_parameters(init_params)

        samples = np.zeros((model.get_nmr_problems(), parameters.shape[1], self.nmr_samples),
                           dtype=np_dtype, order='C')

        self._logger.info('Starting sampling with method {0}'.format(self.__class__.__name__))

        workers = self._create_workers(lambda cl_environment: _MHWorker(
            cl_environment, self.get_compile_flags_list(), model, parameters, samples,
            self.nmr_samples, self.burn_length, self.sample_intervals, self.use_adaptive_proposals))
        self.load_balancer.process(workers, model.get_nmr_problems())

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
            return samples_dict, volume_maps

        return samples_dict

    def _do_initial_logging(self, model):
        self._logger.info('Entered sampling routine.')
        self._logger.info('We will use a {} precision float type for the calculations.'.format(
            'double' if model.double_precision else 'single'))

        for env in self.load_balancer.get_used_cl_environments(self.cl_environments):
            self._logger.info('Using device \'{}\'.'.format(str(env)))

        self._logger.debug('Using compile flags: {}'.format(self.get_compile_flags_list()))

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

    def __init__(self, cl_environment, compile_flags, model, parameters, samples,
                 nmr_samples, burn_length, sample_intervals, use_adaptive_proposals):
        super(_MHWorker, self).__init__(cl_environment)

        self._model = model
        self._parameters = parameters
        self._nmr_params = parameters.shape[1]
        self._samples = samples

        self._nmr_samples = nmr_samples
        self._burn_length = burn_length
        self._sample_intervals = sample_intervals
        self.use_adaptive_proposals = use_adaptive_proposals

        self._workgroup_size = min(cl_environment.device.max_work_group_size,
                                   max(1, 2**math.floor(math.log(self._model.get_nmr_inst_per_problem(), 2))))

        self._rand123_starting_point = RandomStartingPoint()

        self._kernel = self._build_kernel(compile_flags)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        data_buffers = [cl.Buffer(self._cl_run_context.context,
                                  cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                  hostbuf=self._parameters[range_start:range_end, :])]
        samples_buf = cl.Buffer(self._cl_run_context.context,
                                cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                hostbuf=self._samples[range_start:range_end, ...])
        data_buffers.append(samples_buf)

        for data in self._model.get_data():
            data_buffers.append(cl.Buffer(self._cl_run_context.context,
                                          cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data))

        kernel_event = self._kernel.sample(self._cl_run_context.queue,
                                           (int(nmr_problems * self._workgroup_size),),
                                           (int(self._workgroup_size),),
                                           *data_buffers)

        return [self._enqueue_readout(samples_buf, self._samples, 0, nmr_problems, [kernel_event])]

    def _get_kernel_source(self):
        kernel_param_names = ['global mot_float_type* params',
                              'global mot_float_type* samples']
        kernel_param_names.extend(self._model.get_kernel_param_names(self._cl_environment.device))

        proposal_state_size = len(self._model.get_proposal_state())

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + str(self._model.get_nmr_inst_per_problem()) + '''
        '''
        kernel_source += get_random123_cl_code()
        kernel_source += get_float_type_def(self._model.double_precision)
        kernel_source += self._model.get_kernel_data_struct(self._cl_environment.device)
        kernel_source += self._model.get_log_prior_function('getLogPrior', address_space_parameter_vector='local')
        kernel_source += self._model.get_proposal_function('getProposal', address_space_proposal_state='local')
        kernel_source += self._model.get_log_likelihood_per_observation_function('getLogLikelihoodPerObservation',
                                                                                 full_likelihood=False)

        if self.use_adaptive_proposals:
            kernel_source += self._model.get_proposal_state_update_function('updateProposalState',
                                                                            address_space='local')

        if not self._model.is_proposal_symmetric():
            kernel_source += self._model.get_proposal_logpdf('getProposalLogPDF', address_space_proposal_state='local')

        kernel_source += '''
            void _fill_log_likelihood_tmp(const void* const data,
                                          local mot_float_type* const x_local,
                                          local double* log_likelihood_tmp){

                int observation_ind;
                uint local_id = get_local_id(0);
                log_likelihood_tmp[local_id] = 0;

                mot_float_type x_private[''' + str(self._nmr_params) + '''];
                for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    x_private[i] = x_local[i];
                }

                for(int i = 0; i < ceil(NMR_INST_PER_PROBLEM /
                                        (mot_float_type)''' + str(self._workgroup_size) + '''); i++){

                    observation_ind = i * ''' + str(self._workgroup_size) + ''' + local_id;

                    if(observation_ind < NMR_INST_PER_PROBLEM){
                        log_likelihood_tmp[local_id] += getLogLikelihoodPerObservation(data, x_private, observation_ind);
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            void _sum_log_likelihood_tmp_local(local double* log_likelihood_tmp, local double* log_likelihood){
                *log_likelihood = 0;
                for(int i = 0; i < ''' + str(self._workgroup_size) + '''; i++){
                    *log_likelihood += log_likelihood_tmp[i];
                }
            }

            void _sum_log_likelihood_tmp_private(local double* log_likelihood_tmp, private double* log_likelihood){
                *log_likelihood = 0;
                for(int i = 0; i < ''' + str(self._workgroup_size) + '''; i++){
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
                               local mot_float_type* const proposal_state,
                               local uint * const acceptance_counter,
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
                         local mot_float_type* const proposal_state,
                         local uint* const sampling_counter,
                         local uint* const acceptance_counter,
                         global mot_float_type* samples,
                         local double* log_likelihood_tmp){

                uint i, j;
                uint voxel_ind = get_group_id(0);
                bool is_first_work_item = get_local_id(0) == 0;

                for(i = 0; i < ''' + str(self._nmr_samples * (self._sample_intervals + 1)
                                         + self._burn_length) + '''; i++){

                    _update_state(x_local, rng_data, current_likelihood, current_prior,
                                  data, proposal_state, acceptance_counter,
                                  log_likelihood_tmp);

                    if(is_first_work_item){
                        for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                            sampling_counter[j]++;
                        }

                        ''' + ('updateProposalState(proposal_state, sampling_counter, acceptance_counter);'
                               if self.use_adaptive_proposals else '') + '''

                        if(i >= ''' + str(self._burn_length) + '''
                           && i % ''' + str(self._sample_intervals + 1) + ''' == 0){

                            for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                                samples[(uint)((i - ''' + str(self._burn_length) + ''') // remove the burn-in
                                                / ''' + str(self._sample_intervals + 1) + ''') // remove the interval
                                        + j * ''' + str(self._nmr_samples) + '''  // parameter index
                                        + voxel_ind * ''' + str(self._nmr_params * self._nmr_samples) + ''']
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

                uint voxel_ind = get_group_id(0);

                ''' + self._model.get_kernel_data_struct_initialization(self._cl_environment.device,
                                                                        'data', 'voxel_ind') + '''

                rand123_data rand123_rng_data = ''' + self._get_rand123_init_cl_code() + ''';
                void* rng_data = (void*)&rand123_rng_data;

                local mot_float_type x_local[''' + str(self._nmr_params) + '''];

                local double current_likelihood;
                local mot_float_type current_prior;

                local mot_float_type proposal_state[''' + str(proposal_state_size) + '''];
                local uint sampling_counter[''' + str(self._nmr_params) + '''];
                local uint acceptance_counter[''' + str(self._nmr_params) + '''];

                local double log_likelihood_tmp[''' + str(self._workgroup_size) + '''];

                if(get_local_id(0) == 0){
                    for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x_local[i] = params[voxel_ind * ''' + str(self._nmr_params) + ''' + i];
                    }

                    current_prior = getLogPrior((void*)&data, x_local);

                    ''' + ' '.join('proposal_state[{}] = {};'.format(i, v)
                                   for i, v in enumerate(self._model.get_proposal_state())) + '''
                    ''' + ' '.join('sampling_counter[{}] = 0;'.format(i)
                                   for i in range(self._nmr_params)) + '''
                    ''' + ' '.join('acceptance_counter[{}] = 0;'.format(i)
                                   for i in range(self._nmr_params)) + '''
                }
                _fill_log_likelihood_tmp((void*)&data, x_local, log_likelihood_tmp);
                _sum_log_likelihood_tmp_local(log_likelihood_tmp, &current_likelihood);

                _sample(x_local, rng_data, &current_likelihood,
                    &current_prior, (void*)&data, proposal_state, sampling_counter, acceptance_counter,
                    samples, log_likelihood_tmp);
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

