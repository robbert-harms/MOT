import pyopencl as cl
import numpy as np
from mot.cl_routines.mapping.error_measures import ErrorMeasures
from mot.cl_routines.mapping.residual_calculator import ResidualCalculator
from ...utils import results_to_dict, \
    ParameterCLCodeGenerator, initialize_ranlux, get_float_type_def, get_ranlux_cl
from ...load_balance_strategies import Worker
from ...cl_routines.sampling.base import AbstractSampler


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MetropolisHastings(AbstractSampler):

    def __init__(self, cl_environments=None, load_balancer=None, nmr_samples=500, burn_length=500,
                 sample_intervals=5, proposal_update_intervals=50, **kwargs):
        """An CL implementation of Metropolis Hastings.

        Args:
            cl_environments: a list with the cl environments to use
            load_balancer: the load balance strategy to use
            nmr_samples (int): The length of the (returned) chain per voxel
            burn_length (int): The length of the burn in (per voxel), these are extra samples,
                jump is set to 1 (no thinning)
            sample_intervals (int): how many samples we wait before we take,
                this will draw extra samples (chain_length * sample_intervals)
            proposal_update_intervals (int): after how many samples we would like to update the proposals
                This is during burning and sampling. A value of 1 means update after every jump.

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
        self._nmr_samples = nmr_samples or 1
        self.burn_length = burn_length
        self.sample_intervals = sample_intervals
        self.proposal_update_intervals = proposal_update_intervals

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
        var_data_dict = model.get_problems_var_data()
        protocol_data_dict = model.get_problems_protocol_data()
        model_data_dict = model.get_model_data()

        samples = np.zeros((model.get_nmr_problems(), parameters.shape[1], self.nmr_samples),
                           dtype=np_dtype, order='C')

        self._logger.info('Starting sampling with method {0}'.format(self.__class__.__name__))

        workers = self._create_workers(lambda cl_environment: _MHWorker(
            cl_environment, self.get_compile_flags_list(), model, parameters, samples,
            var_data_dict, protocol_data_dict, model_data_dict,
            self.nmr_samples, self.burn_length, self.sample_intervals,
            self.proposal_update_intervals))
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
                               proposal_update_intervals=self.proposal_update_intervals)
        self._logger.info('Sample settings: nmr_samples: {nmr_samples}, burn_length: {burn_length}, '
                          'sample_intervals: {sample_intervals}, '
                          'proposal_update_intervals: {proposal_update_intervals}. '.format(**sample_settings))

        samples_drawn = dict(samples_drawn=(self.burn_length + self.sample_intervals * self.nmr_samples),
                             samples_returned=self.nmr_samples)
        self._logger.info('Total samples drawn: {samples_drawn}, total samples returned: '
                          '{samples_returned} (per problem).'.format(**samples_drawn))


class _MHWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, parameters, samples,
                 var_data_dict, protocol_data_dict, model_data_dict, nmr_samples, burn_length, sample_intervals,
                 proposal_update_intervals):
        super(_MHWorker, self).__init__(cl_environment)

        self._model = model
        self._parameters = parameters
        self._nmr_params = parameters.shape[1]
        self._samples = samples

        self._var_data_dict = var_data_dict
        self._protocol_data_dict = protocol_data_dict
        self._model_data_dict = model_data_dict

        self._nmr_samples = nmr_samples
        self._burn_length = burn_length
        self._sample_intervals = sample_intervals
        self.proposal_update_intervals = proposal_update_intervals

        self._constant_buffers = self._generate_constant_buffers(self._protocol_data_dict, self._model_data_dict)
        self._kernel = self._build_kernel(compile_flags)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        ranluxcltab_buffer = initialize_ranlux(self._cl_run_context, nmr_problems)

        data_buffers = [cl.Buffer(self._cl_run_context.context,
                                  cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                  hostbuf=self._parameters[range_start:range_end, :])]
        samples_buf = cl.Buffer(self._cl_run_context.context,
                                cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                hostbuf=self._samples[range_start:range_end, ...])
        data_buffers.append(samples_buf)
        data_buffers.append(ranluxcltab_buffer)

        for data in self._var_data_dict.values():
            data_buffers.append(cl.Buffer(self._cl_run_context.context,
                                          cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                          hostbuf=data.get_opencl_data()[range_start:range_end, ...]))

        data_buffers.extend(self._constant_buffers)

        kernel_event = self._kernel.sample(self._cl_run_context.queue, (int(nmr_problems), ), None, *data_buffers)

        return [self._enqueue_readout(samples_buf, self._samples, 0, nmr_problems, [kernel_event])]

    def _get_kernel_source(self):
        cl_final_param_transform = self._model.get_final_parameter_transformations('applyFinalParamTransforms')

        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device, self._var_data_dict,
                                                  self._protocol_data_dict, self._model_data_dict)

        kernel_param_names = ['global mot_float_type* params',
                              'global mot_float_type* samples',
                              'global ranluxcl_state_t* ranluxcltab']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        proposal_state_size = len(self._model.get_proposal_state())
        proposal_state = '{' + ', '.join(map(str, self._model.get_proposal_state())) + '}'
        acceptance_counters_between_proposal_updates = '{' + ', '.join('0' * self._nmr_params) + '}'

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + str(self._model.get_nmr_inst_per_problem()) + '''
        '''
        kernel_source += get_ranlux_cl()
        kernel_source += get_float_type_def(self._model.double_precision)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += self._model.get_log_prior_function('getLogPrior')
        kernel_source += self._model.get_proposal_function('getProposal')
        kernel_source += self._model.get_proposal_state_update_function('updateProposalState')

        if cl_final_param_transform:
            kernel_source += cl_final_param_transform

        if not self._model.is_proposal_symmetric():
            kernel_source += self._model.get_proposal_logpdf('getProposalLogPDF')

        kernel_source += self._model.get_log_likelihood_function('getLogLikelihood', full_likelihood=False)

        kernel_source += '''
            void _update_proposals(mot_float_type* const proposal_state, uint* const ac_between_proposal_updates,
                                   uint* const proposal_update_count){

                *proposal_update_count += 1;

                if(*proposal_update_count == ''' + str(self.proposal_update_intervals) + '''){
                    updateProposalState(ac_between_proposal_updates,
                                        ''' + str(self.proposal_update_intervals) + ''',
                                        proposal_state);

                    for(int i = 0; i < ''' + str(proposal_state_size) + '''; i++){
                        ac_between_proposal_updates[i] = 0;
                    }

                    *proposal_update_count = 0;
                }
            }

            void _update_state(mot_float_type* const x,
                               ranluxcl_state_t* ranluxclstate,
                               double* const current_likelihood,
                               mot_float_type* const current_prior,
                               const optimize_data* const data,
                               mot_float_type* const proposal_state,
                               uint * const ac_between_proposal_updates){

                float4 randomnmr;
                mot_float_type new_prior;
                double new_likelihood;
                double bayesian_f;
                mot_float_type old_x;

                #pragma unroll 1
                for(int k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                    randomnmr = ranluxcl32(ranluxclstate);

                    old_x = x[k];
                    x[k] = getProposal(k, x[k], ranluxclstate, proposal_state);

                    new_prior = getLogPrior(x);

                    if(exp(new_prior) > 0){
                        new_likelihood = getLogLikelihood(data, x);
        '''
        if self._model.is_proposal_symmetric():
            kernel_source += '''
                        bayesian_f = exp((new_likelihood + new_prior) - (*current_likelihood + *current_prior));
                '''
        else:
            kernel_source += '''
                        mot_float_type x_to_prop = getProposalLogPDF(k, old_x, x[k], proposal_state);
                        mot_float_type prop_to_x = getProposalLogPDF(k, x[k], x[k], proposal_state);

                        bayesian_f = exp((new_likelihood + new_prior + x_to_prop) -
                            (*current_likelihood + *current_prior + prop_to_x));
                '''
        kernel_source += '''
                        if(randomnmr.x < bayesian_f){
                            *current_likelihood = new_likelihood;
                            *current_prior = new_prior;
                            ac_between_proposal_updates[k]++;
                        }
                        else{
                            x[k] = old_x;
                        }
                    }
                    else{
                        x[k] = old_x;
                    }
                }
            }

            void _sample(mot_float_type* const x,
                         ranluxcl_state_t* ranluxclstate,
                         double* const current_likelihood,
                         mot_float_type* const current_prior,
                         const optimize_data* const data,
                         mot_float_type* const proposal_state,
                         uint* const ac_between_proposal_updates,
                         uint* const proposal_update_count,
                         global mot_float_type* samples){

                uint i, j;
                uint gid = get_global_id(0);
                mot_float_type x_saved[''' + str(self._nmr_params) + '''];

                for(i = 0; i < ''' + str(self._nmr_samples * self._sample_intervals + self._burn_length) + '''; i++){
                    _update_state(x, ranluxclstate, current_likelihood, current_prior,
                                  data, proposal_state, ac_between_proposal_updates);
                    _update_proposals(proposal_state, ac_between_proposal_updates, proposal_update_count);

                    if(i >= ''' + str(self._burn_length) + ''' && i % ''' + str(self._sample_intervals) + ''' == 0){
                        for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                            x_saved[j] = x[j];
                        }

                        ''' + ('applyFinalParamTransforms(data, x_saved);' if cl_final_param_transform else '') + '''

                        for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                            samples[(uint)((i - ''' + str(self._burn_length) + ''')
                                            / ''' + str(self._sample_intervals) + ''')
                                    + j * ''' + str(self._nmr_samples) + '''
                                    + gid * ''' + str(self._nmr_params) + ''' * ''' + str(self._nmr_samples) + ''']
                                        = x_saved[j];
                        }
                    }
                }
            }

            __kernel void sample(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    uint gid = get_global_id(0);
                    uint proposal_update_count = 0;

                    mot_float_type proposal_state[] = ''' + proposal_state + ''';
                    uint ac_between_proposal_updates[] = ''' + acceptance_counters_between_proposal_updates + ''';

                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    ranluxcl_state_t ranluxclstate;
                    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

                    mot_float_type x[''' + str(self._nmr_params) + '''];

                    for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(self._nmr_params) + ''' + i];
                    }

                    double current_likelihood = getLogLikelihood(&data, x);
                    mot_float_type current_prior = getLogPrior(x);

                    _sample(x, &ranluxclstate, &current_likelihood,
                            &current_prior, &data, proposal_state, ac_between_proposal_updates,
                            &proposal_update_count, samples);

                    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
            }
        '''
        return kernel_source
