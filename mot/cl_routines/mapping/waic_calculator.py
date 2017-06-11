import pyopencl as cl
import numpy as np
from ...utils import get_float_type_def, all_logging_disabled
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker
from copy import copy

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class WAICCalculator(CLRoutine):

    def calculate(self, model, samples, evaluation_model=None):
        r"""Compute the Watanabe-Akaike information criterion (WAIC; Watanabe, 2010) for the given samples.

        See http://www.stat.columbia.edu/~gelman/research/unpublished/waic_stan.pdf for a gentle introduction to
        this implementation.

        The WAIC can be used as a model selection criteria, in which the model with the lowest WAIC is preferred.

        Args:
            model (AbstractModel): The model to calculate the WAIC for
            samples (ndarray): The obtained samples per problem. This is supposed to be a matrix
                of shape (d, p, s) with d problems, p parameters and s samples.
            evaluation_model (EvaluationModel): the evaluation model to use for the log likelihood. If not given
                we use the one defined in the model.

        Returns:
            ndarray: per problem the calculated WAIC
        """
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        nmr_problems = samples.shape[0]
        nmr_observations = model.get_nmr_inst_per_problem()

        waics = np.zeros((nmr_problems,))

        samples = np.require(samples, dtype=np_dtype, requirements=['C', 'A', 'O'])
        logsumexps = np.zeros(nmr_observations, dtype=np.float64, order='C')
        variances = np.zeros(nmr_observations, dtype=np.float64, order='C')

        workers = self._create_workers(
            lambda cl_environment: _LLWorker(cl_environment,
                                             self.get_compile_flags_list(model.double_precision),
                                             model, samples, logsumexps, variances, evaluation_model))

        for problem_ind in range(nmr_problems):

            for worker in workers:
                worker.set_problem_index(problem_ind)

            with all_logging_disabled():
                self.load_balancer.process(workers, nmr_observations)

            lpd = np.sum(logsumexps)
            p_waic = np.sum(variances)

            waics[problem_ind] = -2 * (lpd - p_waic)

        return waics


class _LLWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, samples, logsumexps, variances, evaluation_model):
        super(_LLWorker, self).__init__(cl_environment)

        self._model = model
        self._double_precision = model.double_precision
        self._evaluation_model = evaluation_model

        self._samples = samples
        self._log_sum_exps = logsumexps
        self._variances = variances

        self._nmr_samples = self._samples.shape[2]
        self._nmr_params = self._samples.shape[1]
        self._nmr_observations = self._log_sum_exps.shape[0]

        self._problem_index = 0

        self._ll_calculating_buffers, self._ll_buffer, self._lse_buffer, self._variances_buffer = self._create_buffers()

        self._ll_calculating_kernel = self._build_kernel(self._get_ll_calculating_kernel(), compile_flags)
        self._statistic_kernels = self._build_kernel(self._get_statistics_kernels(), compile_flags)

    def set_problem_index(self, problem_index):
        """Set the problem index of the samples we are currently working on."""
        self._problem_index = problem_index

    def calculate(self, range_start, range_end):
        self._calculate_lls(range_start, range_end)
        self._calculate_statistics(range_start, range_end)

        self._enqueue_readout(self._lse_buffer, self._log_sum_exps, range_start, range_end)
        self._enqueue_readout(self._variances_buffer, self._variances, range_start, range_end)

    def _calculate_lls(self, range_start, range_end):
        nmr_problems = range_end - range_start
        buffers = copy(self._ll_calculating_buffers)
        buffers.insert(0, np.uint32(self._problem_index))

        arg_dtypes = [None] * len(buffers)
        arg_dtypes[0] = np.uint32

        kernel = self._ll_calculating_kernel.run_kernel

        kernel.set_scalar_arg_dtypes(arg_dtypes)

        kernel(self._cl_run_context.queue, (int(nmr_problems), int(self._nmr_samples)), None,
                                    *buffers, global_offset=(int(range_start), 0))

    def _calculate_statistics(self, range_start, range_end):
        nmr_problems = range_end - range_start

        max_work_group_sizes = [
            cl.Kernel(self._statistic_kernels, 'mean_and_max').get_work_group_info(
                cl.kernel_work_group_info.WORK_GROUP_SIZE, self._cl_environment.device),
            cl.Kernel(self._statistic_kernels, 'logsum_variance').get_work_group_info(
                cl.kernel_work_group_info.WORK_GROUP_SIZE, self._cl_environment.device)
        ]
        workgroup_size = max(max_work_group_sizes)

        lse_tmp_buffer = cl.LocalMemory(workgroup_size * np.dtype('double').itemsize)
        var_tmp_buffer = cl.LocalMemory(workgroup_size * np.dtype('double').itemsize)

        buffers = [self._ll_buffer, self._lse_buffer, self._variances_buffer, lse_tmp_buffer, var_tmp_buffer]

        self._statistic_kernels.mean_and_max(
            self._cl_run_context.queue, (int(nmr_problems * workgroup_size),), (int(workgroup_size),),
            *buffers, global_offset=(int(range_start * workgroup_size),))

        self._statistic_kernels.logsum_variance(
            self._cl_run_context.queue, (int(nmr_problems * workgroup_size),), (int(workgroup_size),),
            *buffers, global_offset=(int(range_start * workgroup_size),))

    def _create_buffers(self):
        ll_buffer = cl.Buffer(self._cl_run_context.context,
                              cl.mem_flags.READ_WRITE,
                              size=(np.dtype(np.float64).itemsize * self._nmr_observations * self._nmr_samples))

        lse_buffer = cl.Buffer(self._cl_run_context.context,
                               cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                               hostbuf=self._log_sum_exps)

        variances_buffer = cl.Buffer(self._cl_run_context.context,
                                     cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                     hostbuf=self._variances)

        samples_buffer = cl.Buffer(self._cl_run_context.context,
                                   cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                   hostbuf=self._samples)

        ll_calculating_buffers = [samples_buffer, ll_buffer]

        for data in self._model.get_data():
            ll_calculating_buffers.append(cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data))

        return ll_calculating_buffers, ll_buffer, lse_buffer, variances_buffer

    def _get_ll_calculating_kernel(self):
        """Kernel to calculate the log likelihoods per sample per observation"""
        cl_func = self._model.get_log_likelihood_per_observation_function(
            'getLogLikelihoodPerObs', evaluation_model=self._evaluation_model, full_likelihood=True)

        kernel_param_names = ['uint problem_ind', 'global mot_float_type* samples', 'global double* lls']
        kernel_param_names.extend(self._model.get_kernel_param_names(self._cl_environment.device))
        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._model.get_kernel_data_struct(self._cl_environment.device)
        kernel_source += cl_func
        kernel_source += r'''
            __kernel void run_kernel(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ulong obs_ind = get_global_id(0);
                    ulong sample_ind = get_global_id(1);

                    ''' + self._model.get_kernel_data_struct_initialization(self._cl_environment.device,
                                                                            'data', problem_id_name='problem_ind') + '''

                    mot_float_type x[''' + str(self._nmr_params) + '''];

                    for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x[i] = samples[problem_ind * ''' + str(self._nmr_params * self._nmr_samples) + ''' +
                                       i * ''' + str(self._nmr_samples) + ''' + sample_ind];
                    }

                    lls[obs_ind * ''' + str(self._nmr_samples) + ''' + sample_ind] =
                        getLogLikelihoodPerObs((void*)&data, x, obs_ind);

                }
        '''
        return kernel_source

    def _get_statistics_kernels(self):
        kernel_param_names = ['global double* lls', 'global double* lses', 'global double* variances',
                              'local double* lse_tmp', 'local double* var_tmp']
        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += r'''
            __kernel void mean_and_max(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ulong obs_ind = (ulong)(get_global_id(0) / get_local_size(0));
                    ulong local_id = get_local_id(0);
                    uint workgroup_size = get_local_size(0);

                    ulong sample_ind;

                    double ll;
                    double max_ll = -HUGE_VAL;
                    double mean_sum = 0;

                    for(uint i = 0; i < ceil(''' + str(self._nmr_samples) + ''' / (mot_float_type)workgroup_size); i++){
                        sample_ind = i * workgroup_size + local_id;

                        if(sample_ind < ''' + str(self._nmr_samples) + '''){
                            ll = lls[obs_ind * ''' + str(self._nmr_samples) + ''' + sample_ind];

                            max_ll = max(max_ll, ll);
                            mean_sum += ll;
                        }
                    }
                    lse_tmp[local_id] = max_ll;
                    var_tmp[local_id] = mean_sum;

                    barrier(CLK_LOCAL_MEM_FENCE);

                    if(local_id == 0){
                        mean_sum = 0;
                        for(uint i = 0; i < workgroup_size; i++){
                            max_ll = max(max_ll, lse_tmp[i]);
                            mean_sum += var_tmp[i];
                        }
                        lses[obs_ind] = max_ll;
                        variances[obs_ind] = mean_sum / ''' + str(self._nmr_samples) + ''';
                    }

                }

            __kernel void logsum_variance(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ulong obs_ind = (ulong)(get_global_id(0) / get_local_size(0));
                    ulong sample_ind;
                    ulong local_id = get_local_id(0);
                    uint workgroup_size = get_local_size(0);

                    double ll;
                    double max_ll = lses[obs_ind];
                    double mean_ll = variances[obs_ind];
                    double exp_sum = 0;
                    double var_sum = 0;

                    for(uint i = 0; i < ceil(''' + str(self._nmr_samples) + ''' / (mot_float_type)workgroup_size); i++){
                        sample_ind = i * workgroup_size + local_id;

                        if(sample_ind < ''' + str(self._nmr_samples) + '''){
                            ll = lls[obs_ind * ''' + str(self._nmr_samples) + ''' + sample_ind];

                            exp_sum += exp(ll - max_ll);
                            var_sum += pown(ll - mean_ll, 2);
                        }
                    }
                    lse_tmp[local_id] = exp_sum;
                    var_tmp[local_id] = var_sum;

                    barrier(CLK_LOCAL_MEM_FENCE);

                    if(local_id == 0){
                        exp_sum = 0;
                        var_sum = 0;
                        for(uint i = 0; i < workgroup_size; i++){
                            exp_sum += lse_tmp[i];
                            var_sum += var_tmp[i];
                        }

                        lses[obs_ind] += log(exp_sum / ''' + str(self._nmr_samples) + ''');
                        variances[obs_ind] = var_sum / (''' + str(self._nmr_samples) + ''' - 1);
                    }
                }
        '''
        return kernel_source
