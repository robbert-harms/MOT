import pyopencl as cl
import numpy as np
from ...cl_functions import RanluxCL
from ...utils import get_cl_pragma_double, results_to_dict, \
    ParameterCLCodeGenerator, initialize_ranlux, get_float_type_def
from ...load_balance_strategies import Worker
from ...cl_routines.sampling.base import AbstractSampler


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MetropolisHastings(AbstractSampler):

    def __init__(self, cl_environments, load_balancer, nmr_samples=500, burn_length=1000,
                 sample_intervals=5, proposal_update_intervals=30):
        """An CL implementation of Metropolis Hastings.

        Args:
            cl_environments: a list with the cl environments to use
            load_balancer: the load balance strategy to use
            nmr_samples (int): The length of the (returned) chain per voxel
            burn_length (int): The length of the burn in (per voxel), these are extra samples,
                jump is set to 1 (no thinning)
            sample_intervales (int): how many samples we wait before we take,
                this requires extra samples (chain_length * sample_intervals)
            proposal_update_intervals (int): after how many samples we would like to update the proposals
                This is during burning and sampling. A value of 1 means update after every jump.

        Attributes:
            nmr_samples (int): The length of the (returned) chain per voxel
            burn_length (int): The length of the burn in (per voxel), these are extra samples,
                jump is set to 1 (no thinning)
            sample_intervales (int): how many samples we wait before we take,
                this requires extra samples (chain_length * sample_intervals)
            proposal_update_intervals (int): after how many samples we would like to update the proposals
                This is during burning and sampling. A value of 1 means update after every jump.
        """

        super(MetropolisHastings, self).__init__(cl_environments=cl_environments, load_balancer=load_balancer)
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
        parameters = model.get_initial_parameters(init_params).astype(np.float64, order='C')
        var_data_dict = model.get_problems_var_data()
        prtcl_data_dict = model.get_problems_prtcl_data()
        fixed_data_dict = model.get_problems_fixed_data()

        samples = np.zeros((model.get_nmr_problems(), parameters.shape[1] * self.nmr_samples),
                           dtype=np.float64, order='C')

        workers = self._create_workers(_MHWorker, model, parameters, samples,
                                       var_data_dict, prtcl_data_dict, fixed_data_dict,
                                       self.nmr_samples, self.burn_length, self.sample_intervals,
                                       self.proposal_update_intervals)
        self.load_balancer.process(workers, model.get_nmr_problems())

        samples = np.reshape(samples, (model.get_nmr_problems(), parameters.shape[1], self.nmr_samples))
        samples_dict = results_to_dict(samples, model.get_optimized_param_names())

        if full_output:
            volume_maps = model.finalize_optimization_results(model.samples_to_statistics(samples_dict))
            return samples_dict, {'volume_maps': volume_maps}
        return samples_dict


class _MHWorker(Worker):

    def __init__(self, cl_environment, model, parameters, samples,
                 var_data_dict, prtcl_data_dict, fixed_data_dict, nmr_samples, burn_length, sample_intervals,
                 proposal_update_intervals):
        super(_MHWorker, self).__init__(cl_environment)

        self._model = model
        self._use_double = model.use_double
        self._parameters = parameters
        self._nmr_params = parameters.shape[1]
        self._samples = samples

        self._var_data_dict = var_data_dict
        self._prtcl_data_dict = prtcl_data_dict
        self._fixed_data_dict = fixed_data_dict

        self._nmr_samples = nmr_samples
        self._burn_length = burn_length
        self._sample_intervals = sample_intervals
        self.proposal_update_intervals = proposal_update_intervals

        self._constant_buffers = self._generate_constant_buffers(self._prtcl_data_dict, self._fixed_data_dict)
        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        ranluxcltab_buffer = initialize_ranlux(self._cl_environment, self._queue, nmr_problems)

        read_only_flags = self._cl_environment.get_read_only_cl_mem_flags()
        write_only_flags = self._cl_environment.get_write_only_cl_mem_flags()

        data_buffers = [cl.Buffer(self._cl_environment.context, read_only_flags,
                                  hostbuf=self._parameters[range_start:range_end, :])]
        samples_buf = cl.Buffer(self._cl_environment.context, write_only_flags,
                                hostbuf=self._samples[range_start:range_end, :])
        data_buffers.append(samples_buf)

        data_buffers.append(ranluxcltab_buffer)
        data_buffers.append(np.uint32(self._nmr_samples))
        data_buffers.append(np.uint32(self._burn_length))
        data_buffers.append(np.uint32(self._sample_intervals))
        data_buffers.append(np.uint32(self._nmr_params))
        # We add these parameters to the kernel call instead of inlining them.
        # This if for a good reason. At the time of writing, the CL kernel compiler crashes if these parameters
        # are inlined in the source file.
        # My guess is that the compiler tries to optimize the loops, fails and then crashes.

        for data in self._var_data_dict.values():
            if len(data.shape) < 2:
                data_buffers.append(cl.Buffer(self._cl_environment.context,
                                              read_only_flags, hostbuf=data[range_start:range_end]))
            else:
                data_buffers.append(cl.Buffer(self._cl_environment.context, read_only_flags,
                                              hostbuf=data[range_start:range_end, :]))

        data_buffers.extend(self._constant_buffers)

        global_range = (int(nmr_problems), )
        local_range = None

        event = self._kernel.sample(self._queue, global_range, local_range, *data_buffers)
        event = cl.enqueue_copy(self._queue, self._samples[range_start:range_end, :],
                                samples_buf, wait_for=(event,), is_blocking=False)
        return event

    def _get_kernel_source(self):
        #todo, this accepts a model_float now, we have to change the code to allow this to work
        cl_final_param_transform = self._model.get_final_parameter_transformations('applyFinalParamTransforms')

        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device, self._var_data_dict,
                                                  self._prtcl_data_dict, self._fixed_data_dict)

        kernel_param_names = ['global double* params',
                              'global double* samples',
                              'global float4* ranluxcltab',
                              'unsigned int nmr_samples',
                              'unsigned int burn_length',
                              'unsigned int sample_intervals',
                              'unsigned int nmr_params']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        rng_code = RanluxCL()

        nrm_adaptable_proposal_parameters = len(self._model.get_proposal_parameter_values())
        adaptable_proposal_parameters_str = '{' + ', '.join(map(str, self._model.get_proposal_parameter_values())) + '}'
        acceptance_counters_between_proposal_updates = '{' + ', '.join('0' * self._nmr_params) + '}'

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + str(self._model.get_nmr_inst_per_problem()) + '''
        '''
        kernel_source += get_cl_pragma_double()
        kernel_source += get_float_type_def(self._use_double)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += rng_code.get_cl_header()
        kernel_source += self._model.get_log_prior_function('getLogPrior')
        kernel_source += self._model.get_proposal_function('getProposal')
        kernel_source += self._model.get_proposal_parameters_update_function('updateProposalParameters')

        if cl_final_param_transform:
            kernel_source += cl_final_param_transform

        if not self._model.is_proposal_symmetric():
            kernel_source += self._model.get_proposal_logpdf('getProposalLogPDF')

        kernel_source += self._model.get_log_likelihood_function('getLogLikelihood')

        kernel_source += '''
            void update_state(double* const x,
                              double* const x_proposal,
                              ranluxcl_state_t* ranluxclstate,
                              double* const current_likelihood,
                              double* const current_prior,
                              const optimize_data* const data,
                              double * const proposal_parameters,
                              uint * const ac_between_proposal_updates,
                              const uint nmr_params){

                float4 randomnmr;
                double new_prior;
                double new_likelihood;
                double bayesian_f;
        '''
        if not self._model.is_proposal_symmetric():
            kernel_source += '''
                double x_to_prop, prop_to_x;
        '''
        kernel_source += '''
                for(int k = 0; k < nmr_params; k++){
                    randomnmr = ranluxcl(ranluxclstate);

                    for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x_proposal[i] = x[i];
                    }
                    x_proposal[k] = getProposal(k, x[k], ranluxclstate, proposal_parameters);

                    new_prior = getLogPrior(x_proposal);

                    if(exp(new_prior) > 0){
                        new_likelihood = getLogLikelihood(data, x_proposal);

        '''
        if self._model.is_proposal_symmetric():
            kernel_source += '''
                        bayesian_f = exp((new_likelihood + new_prior) - (*current_likelihood + *current_prior));
                '''
        else:
            kernel_source += '''
                        x_to_prop = getProposalLogPDF(k, x[k], x_proposal[k], proposal_parameters);
                        prop_to_x = getProposalLogPDF(k, x_proposal[k], x[k], proposal_parameters);

                        bayesian_f = exp((new_likelihood + new_prior + x_to_prop) -
                            (*current_likelihood + *current_prior + prop_to_x));
                '''
        kernel_source += '''
                        if(randomnmr.x < bayesian_f){
                            for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                                x[i] = x_proposal[i];
                            }

                            *current_likelihood = new_likelihood;
                            *current_prior = new_prior;
                            ac_between_proposal_updates[k]++;
                        }
                    }
                }
            }
        '''
        kernel_source += '''
            void _update_proposals(double* const proposal_parameters, uint* const ac_between_proposal_updates){
                updateProposalParameters(ac_between_proposal_updates, ''' + str(self.proposal_update_intervals) + ''',
                                         proposal_parameters);

                for(int i = 0; i < ''' + str(nrm_adaptable_proposal_parameters) + '''; i++){
                    ac_between_proposal_updates[i] = 0;
                }
            }
        '''
        kernel_source += '''
            __kernel void sample(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    unsigned int i, j, proposal_update_count;

                    double proposal_parameters[] = ''' + adaptable_proposal_parameters_str + ''';
                    uint ac_between_proposal_updates[] = ''' + acceptance_counters_between_proposal_updates + ''';

                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    ranluxcl_state_t ranluxclstate;
                    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

                    double x[''' + str(self._nmr_params) + '''];
                    double x_proposal[''' + str(self._nmr_params) + '''];

                    for(i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(self._nmr_params) + ''' + i];
                    }

                    double current_likelihood = getLogLikelihood(&data, x);
                    double current_prior = getLogPrior(x);

                    proposal_update_count = 0;
                    for(i = 0; i < burn_length; i++){
                        update_state(x, x_proposal, &ranluxclstate, &current_likelihood,
                                     &current_prior, &data, proposal_parameters, ac_between_proposal_updates,
                                     nmr_params);

                        proposal_update_count += 1;
                        if(proposal_update_count == ''' + str(self.proposal_update_intervals) + '''){
                            _update_proposals(proposal_parameters, ac_between_proposal_updates);
                            proposal_update_count = 0;
                        }
                    }

                    proposal_update_count = 0;
                    for(i = 0; i < nmr_samples; i++){
                        for(j = 0; j < sample_intervals; j++){
                            update_state(x, x_proposal, &ranluxclstate, &current_likelihood,
                                         &current_prior, &data, proposal_parameters, ac_between_proposal_updates,
                                         nmr_params);

                            proposal_update_count += 1;
                            if(proposal_update_count == ''' + str(self.proposal_update_intervals) + '''){
                                _update_proposals(proposal_parameters, ac_between_proposal_updates);
                                proposal_update_count = 0;
                            }
                        }

                        for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                            x_proposal[j] = x[j];
                        }

                        ''' + ('applyFinalParamTransforms(&data, x_proposal);'
                               if cl_final_param_transform else '') + '''

                        for(j = 0; j < ''' + str(self._nmr_params) + '''; j++){
                            samples[i + j * nmr_samples + gid *
                                    ''' + str(self._nmr_params) + ''' * nmr_samples] = x_proposal[j];
                        }
                    }

                    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
            }
        '''
        kernel_source += rng_code.get_cl_code()
        return kernel_source