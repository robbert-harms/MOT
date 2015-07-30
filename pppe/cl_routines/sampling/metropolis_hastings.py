import pyopencl as cl
import numpy as np
from ...cl_functions import RanluxCL
from ...utils import get_cl_double_extension_definer, results_to_dict, get_read_only_cl_mem_flags, \
    get_write_only_cl_mem_flags, set_correct_cl_data_type, ParameterCLCodeGenerator, \
    initialize_ranlux
from ...load_balance_strategies import Worker
from ...cl_routines.sampling.base import AbstractSampler


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MetropolisHastings(AbstractSampler):

    def __init__(self, cl_environments, load_balancer, nmr_samples=100, burn_length=0, sample_intervals=1):
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
        var_data_dict = set_correct_cl_data_type(model.get_problems_var_data())
        prtcl_data_dict = set_correct_cl_data_type(model.get_problems_prtcl_data())
        fixed_data_dict = set_correct_cl_data_type(model.get_problems_fixed_data())

        samples = np.zeros((model.get_nmr_problems(), parameters.shape[1] * self.nmr_samples),
                           dtype=np.float64, order='C')
        acceptance_counter = np.zeros((model.get_nmr_problems(), parameters.shape[1]),
                                      dtype=np.uint32, order='C')

        workers = self._create_workers(_MHWorker, model, parameters, samples, acceptance_counter,
                                       var_data_dict, prtcl_data_dict, fixed_data_dict,
                                       self.nmr_samples, self.burn_length, self.sample_intervals)
        self.load_balancer.process(workers, model.get_nmr_problems())

        samples = np.reshape(samples, (model.get_nmr_problems(), parameters.shape[1], self.nmr_samples))
        samples_dict = results_to_dict(samples, model.get_optimized_param_names())

        if full_output:
            steps = (self.nmr_samples * self.sample_intervals + self.burn_length) * parameters.shape[0]

            acceptance_counter = acceptance_counter.astype(np.float32) / float(steps)

            extra_output = {name + '.acceptance_rate': acceptance_counter[:, ind] for ind, name
                            in enumerate(model.get_optimized_param_names())}

            return samples_dict, extra_output
        return samples_dict


class _MHWorker(Worker):

    def __init__(self, cl_environment, model, parameters, samples, acceptance_counter,
                 var_data_dict, prtcl_data_dict, fixed_data_dict, nmr_samples, burn_length, sample_intervals):
        super(_MHWorker, self).__init__(cl_environment)

        self._model = model
        self._parameters = parameters
        self._nmr_params = parameters.shape[1]
        self._samples = samples
        self._acceptance_counter = acceptance_counter

        self._var_data_dict = var_data_dict
        self._prtcl_data_dict = prtcl_data_dict
        self._fixed_data_dict = fixed_data_dict

        self._nmr_samples = nmr_samples
        self._burn_length = burn_length
        self._sample_intervals = sample_intervals

        self._constant_buffers = self._generate_constant_buffers(self._prtcl_data_dict, self._fixed_data_dict)
        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        ranluxcltab_buffer = initialize_ranlux(self._cl_environment, self._queue, nmr_problems)

        read_only_flags = get_read_only_cl_mem_flags(self._cl_environment)
        write_only_flags = get_write_only_cl_mem_flags(self._cl_environment)

        data_buffers = [cl.Buffer(self._cl_environment.context, read_only_flags,
                                  hostbuf=self._parameters[range_start:range_end, :])]
        samples_buf = cl.Buffer(self._cl_environment.context, write_only_flags,
                                hostbuf=self._samples[range_start:range_end, :])
        data_buffers.append(samples_buf)

        ac_buf = cl.Buffer(self._cl_environment.context, write_only_flags,
                           hostbuf=self._acceptance_counter[range_start:range_end, :])
        data_buffers.append(ac_buf)

        data_buffers.append(ranluxcltab_buffer)
        data_buffers.append(np.uint32(self._nmr_samples))
        data_buffers.append(np.uint32(self._burn_length))
        data_buffers.append(np.uint32(self._sample_intervals))

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
        event = cl.enqueue_copy(self._queue, self._acceptance_counter[range_start:range_end, :],
                                ac_buf, wait_for=(event,), is_blocking=False)
        event = cl.enqueue_copy(self._queue, self._samples[range_start:range_end, :],
                                samples_buf, wait_for=(event,), is_blocking=False)
        return event

    def _get_kernel_source(self):
        cl_final_param_transform = self._model.get_final_parameter_transformations('applyFinalParamTransforms')

        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device, self._var_data_dict,
                                                  self._prtcl_data_dict, self._fixed_data_dict)

        kernel_param_names = ['global double* params',
                              'global double* samples',
                              'global unsigned int* acceptance_counter',
                              'global float4* ranluxcltab',
                              'unsigned int nmr_samples',
                              'unsigned int burn_length',
                              'unsigned int sample_intervals']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        rng_code = RanluxCL()

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + repr(self._model.get_nmr_inst_per_problem()) + '''
        '''
        kernel_source += get_cl_double_extension_definer(self._cl_environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += rng_code.get_cl_header()
        kernel_source += self._model.get_log_prior_function('getLogPrior')
        kernel_source += self._model.get_proposal_function('getProposal')

        if cl_final_param_transform:
            kernel_source += cl_final_param_transform

        if not self._model.is_proposal_symmetric():
            kernel_source += self._model.get_proposal_logpdf('getProposalLogPDF')

        kernel_source += self._model.get_log_likelihood_function('getLogLikelihood')

        kernel_source += '''
            void update_state(double* const x,
                              double* const x_proposal,
                              global int* const acceptance_counter,
                              ranluxcl_state_t* ranluxclstate,
                              double* const current_likelihood,
                              double* const current_prior,
                              const optimize_data* const data){

                for(int k = 0; k < ''' + repr(self._nmr_params) + '''; k++){
                    float4 randomnmr = ranluxcl(ranluxclstate);

                    for(int i = 0; i < ''' + repr(self._nmr_params) + '''; i++){
                        x_proposal[i] = x[i];
                    }
                    x_proposal[k] = getProposal(k, x[k], ranluxclstate);

                    double new_prior = getLogPrior(x_proposal);

                    if(exp(new_prior) > 0){
                        double new_likelihood = getLogLikelihood(data, x_proposal);

        '''
        if self._model.is_proposal_symmetric():
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
                            for(int i = 0; i < ''' + repr(self._nmr_params) + '''; i++){
                                x[i] = x_proposal[i];
                            }

                            *current_likelihood = new_likelihood;
                            *current_prior = new_prior;
                            acceptance_counter[k + ''' + repr(self._nmr_params) + ''' * get_global_id(0)]++;
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
                    unsigned int i, j;

                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    ranluxcl_state_t ranluxclstate;
                    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

                    double x[''' + repr(self._nmr_params) + '''];
                    double x_proposal[''' + repr(self._nmr_params) + '''];

                    for(i = 0; i < ''' + repr(self._nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + repr(self._nmr_params) + ''' + i];
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

                        for(j = 0; j < ''' + repr(self._nmr_params) + '''; j++){
                            x_proposal[j] = x[j];
                        }
                        ''' + ('applyFinalParamTransforms(&data, x_proposal);'
                               if cl_final_param_transform else '') + '''
                        for(j = 0; j < ''' + repr(self._nmr_params) + '''; j++){
                            samples[i + j * nmr_samples + gid * ''' + repr(self._nmr_params) +\
                                    ''' * nmr_samples] = x_proposal[j];
                        }
                    }

                    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
            }
        '''
        kernel_source += rng_code.get_cl_code()
        return kernel_source