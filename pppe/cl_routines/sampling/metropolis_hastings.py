import warnings

import pyopencl as cl
import numpy as np

from ...cl_functions import RanluxCL
from ...utils import get_cl_double_extension_definer, results_to_dict, get_read_only_cl_mem_flags, \
    get_write_only_cl_mem_flags, set_correct_cl_data_type, ParameterCLCodeGenerator, \
    initialize_ranlux
from ...load_balance_strategies import WorkerConstructor
from ...cl_routines.sampling.base import AbstractSampler


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MetropolisHastings(AbstractSampler):

    def __init__(self, cl_environments=None, load_balancer=None,
                 nmr_samples=500, burn_length=500, sample_intervals=10):
        """An CL implementation of Metropolis Hastings.

        Args:
            cl_environments: a list with the cl environments to use
            load_balancer: the load balance strategy to use
            nmr_samples (int): The length of the (returned) chain per voxel
            burn_length (int): The length of the burn in (per voxel), these are extra samples,
                jump is set to 1 (no thinning)
            sample_intervales (int): how many samples we wait before we take,
                this requires extra samples (chain_length * sample_intervals)

        Attributes:
            nmr_samples (int): The length of the (returned) chain per voxel
            burn_length (int): The length of the burn in (per voxel), these are extra samples,
                jump is set to 1 (no thinning)
            sample_intervales (int): how many samples we wait before we take,
                this requires extra samples (chain_length * sample_intervals)
        """

        super(MetropolisHastings, self).__init__(cl_environments=cl_environments, load_balancer=load_balancer)
        self._nmr_samples = nmr_samples or 1
        self.burn_length = burn_length
        self.sample_intervals = sample_intervals

    @property
    def nmr_samples(self):
        return self._nmr_samples

    @nmr_samples.setter
    def nmr_samples(self, value):
        self._nmr_samples = value or 1

    def sample(self, model, init_params=None, full_output=False):
        parameters = set_correct_cl_data_type(model.get_initial_parameters(init_params))
        samples_host = np.zeros((model.get_nmr_problems(), parameters.shape[1] * self.nmr_samples),
                                dtype=np.float64, order='C')
        acceptance_counter_host = np.zeros((model.get_nmr_problems(),), dtype=np.int32, order='C')

        var_data_dict = set_correct_cl_data_type(model.get_problems_var_data())
        prtcl_data_dict = set_correct_cl_data_type(model.get_problems_prtcl_data())
        fixed_data_dict = set_correct_cl_data_type(model.get_problems_fixed_data())

        def kernel_source_generator(cl_environment):
            return self._get_kernel_source(model.get_log_likelihood_function('getLogLikelihood'),
                                           model.get_log_prior_function('getLogPrior'),
                                           model.get_proposal_function('getProposal'),
                                           model.is_proposal_symmetric(),
                                           model.get_proposal_logpdf('getProposalLogPDF'),
                                           model.get_final_parameter_transformations('applyFinalParamTransforms'),
                                           parameters.shape[1],
                                           model.get_nmr_inst_per_problem(),
                                           var_data_dict,
                                           prtcl_data_dict,
                                           fixed_data_dict,
                                           cl_environment)

        def minimizer_generator(cl_environment, start, end, buffered_dicts):
            warnings.simplefilter("ignore")

            prtcl_dbuf = buffered_dicts[0]
            fixed_dbuf = buffered_dicts[1]

            kernel_source = kernel_source_generator(cl_environment)
            kernel = cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))

            return self._run_sampler(parameters, samples_host, acceptance_counter_host,
                                     prtcl_dbuf, var_data_dict, fixed_dbuf,
                                     start, end, cl_environment, kernel)

        worker_constructor = WorkerConstructor()
        workers = worker_constructor.generate_workers(self.load_balancer.get_used_cl_environments(self.cl_environments),
                                                      minimizer_generator,
                                                      data_dicts_to_buffer=(prtcl_data_dict,
                                                                            fixed_data_dict))

        self.load_balancer.process(workers, model.get_nmr_problems())

        samples = np.reshape(samples_host, (model.get_nmr_problems(), parameters.shape[1], self.nmr_samples))
        samples_dict = results_to_dict(samples, model.get_optimized_param_names())

        if full_output:
            steps = (self.nmr_samples * self.sample_intervals + self.burn_length) * parameters.shape[0]
            print((acceptance_counter_host / float(steps))[2000])
            return samples_dict, {'ar': acceptance_counter_host / float(steps)}
        return samples_dict

    def _run_sampler(self, parameters, samples_host, acceptance_counter_host,
                     prtcl_data_buffers, var_data_dict, fixed_data_buffers, start, end, cl_environment, kernel):

        queue = cl_environment.get_new_queue()
        nmr_problems = end - start
        ranluxcltab_buffer = initialize_ranlux(cl_environment, queue, nmr_problems)

        read_only_flags = get_read_only_cl_mem_flags(cl_environment)
        write_only_flags = get_write_only_cl_mem_flags(cl_environment)

        data_buffers = [cl.Buffer(cl_environment.context, read_only_flags, hostbuf=parameters[start:end, :])]
        samples_buf = cl.Buffer(cl_environment.context, write_only_flags, hostbuf=samples_host[start:end, :])
        data_buffers.append(samples_buf)

        ac_buf = cl.Buffer(cl_environment.context, write_only_flags, hostbuf=acceptance_counter_host[start:end])
        data_buffers.append(ac_buf)

        data_buffers.append(ranluxcltab_buffer)
        data_buffers.append(np.int32(self.nmr_samples))
        data_buffers.append(np.int32(self.burn_length))
        data_buffers.append(np.int32(self.sample_intervals))

        for data in var_data_dict.values():
            if len(data.shape) < 2:
                data_buffers.append(cl.Buffer(cl_environment.context, read_only_flags, hostbuf=data[start:end]))
            else:
                data_buffers.append(cl.Buffer(cl_environment.context, read_only_flags, hostbuf=data[start:end, :]))

        data_buffers.extend(prtcl_data_buffers)
        data_buffers.extend(fixed_data_buffers)

        global_range = (int(nmr_problems), )
        local_range = None

        event = kernel.sample(queue, global_range, local_range, *data_buffers)
        event = cl.enqueue_copy(queue, acceptance_counter_host[start:end], ac_buf, wait_for=(event,), is_blocking=False)
        event = cl.enqueue_copy(queue, samples_host[start:end, :], samples_buf, wait_for=(event,), is_blocking=False)

        return queue, event

    def _get_kernel_source(self, cl_log_likelihood, cl_prior_func, cl_proposal_func, proposal_is_symmetric,
                           cl_proposal_pdf, cl_final_param_transform, nmr_params,
                           nmr_inst_per_problem, var_data_dict, prtcl_data_dict, model_data_dict, environment):

        param_code_gen = ParameterCLCodeGenerator(environment.device, var_data_dict, prtcl_data_dict, model_data_dict)

        kernel_param_names = ['global double* params',
                              'global double* samples',
                              'global int* acceptance_counter',
                              'global float4* ranluxcltab',
                              'int nmr_samples',
                              'int burn_length',
                              'int sample_intervals']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        rng_code = RanluxCL()

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + repr(nmr_inst_per_problem) + '''
        '''
        kernel_source += get_cl_double_extension_definer(environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += rng_code.get_cl_header()
        kernel_source += cl_prior_func
        kernel_source += cl_proposal_func
        if cl_final_param_transform:
            kernel_source += cl_final_param_transform
        if not proposal_is_symmetric:
            kernel_source += cl_proposal_pdf
        kernel_source += cl_log_likelihood
        kernel_source += '''
            void update_state(double* const x,
                              double* const x_proposal,
                              global int* const acceptance_counter,
                              ranluxcl_state_t* ranluxclstate,
                              double* const current_likelihood,
                              double* const current_prior,
                              const optimize_data* const data){

                for(int k = 0; k < ''' + repr(nmr_params) + '''; k++){
                    float4 randomnmr = ranluxcl(ranluxclstate);

                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        x_proposal[i] = x[i];
                    }
                    x_proposal[k] = getProposal(k, x[k], ranluxclstate);

                    double new_prior = getLogPrior(x_proposal);

                    if(exp(new_prior) > 0){
                        double new_likelihood = getLogLikelihood(data, x_proposal);

        '''
        if proposal_is_symmetric:
            kernel_source += "\t" * 4
            kernel_source += 'double f = exp((new_likelihood + new_prior) - (*current_likelihood + *current_prior));'
        else:
            kernel_source += "\t" * 4 + 'double x_to_prop = getProposalLogPDF(k, x[k], x_proposal[k]);' + "\n"
            kernel_source += "\t" * 4 + 'double prop_to_x = getProposalLogPDF(k, x_proposal[k], x[k]);' + "\n"
            kernel_source += "\t" * 4 + \
                'double f = exp((new_likelihood + new_prior + x_to_prop) - ' \
                '(*current_likelihood + *current_prior + prop_to_x));'

        kernel_source += '''
                        if(randomnmr.x < f){
                            for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                                x[i] = x_proposal[i];
                            }

                            *current_likelihood = new_likelihood;
                            *current_prior = new_prior;
                            acceptance_counter[get_global_id(0)]++;
                        }
                    }
                }
            }
        '''
        kernel_source += '''
            __kernel void sample(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    int i, j;

                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    ranluxcl_state_t ranluxclstate;
                    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

                    double x[''' + repr(nmr_params) + '''];
                    double x_proposal[''' + repr(nmr_params) + '''];

                    for(i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + repr(nmr_params) + ''' + i];
                    }

                    double current_likelihood = getLogLikelihood(&data, x);
                    double current_prior = getLogPrior(x);

                    for(i = 0; i < burn_length; i++){
                        update_state(x, x_proposal, acceptance_counter, &ranluxclstate, &current_likelihood,
                                     &current_prior, &data);
                    }

                    for(i = 0; i < nmr_samples; i++){
                        for(j = 0; j < sample_intervals; j++){
                            update_state(x, x_proposal, acceptance_counter, &ranluxclstate, &current_likelihood,
                                         &current_prior, &data);
                        }

                        for(j = 0; j < ''' + repr(nmr_params) + '''; j++){
                            x_proposal[j] = x[j];
                        }
                        ''' + ('applyFinalParamTransforms(&data, x_proposal);'
                               if cl_final_param_transform else '') + '''
                        for(j = 0; j < ''' + repr(nmr_params) + '''; j++){
                            samples[i + j * nmr_samples + gid * ''' + repr(nmr_params) +\
                                    ''' * nmr_samples] = x_proposal[j];
                        }
                    }

                    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
            }
        '''
        kernel_source += rng_code.get_cl_code()
        return kernel_source