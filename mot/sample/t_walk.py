import numpy as np

from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Struct, Array, LocalMemory
from mot.lib.utils import parse_cl_function
from mot.sample.base import AbstractSampler

__author__ = 'Robbert Harms'
__date__ = '2018-10-20'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class ThoughtfulWalk(AbstractSampler):

    def __init__(self, ll_func, log_prior_func, x0, x1, finalize_proposal_func=None,
                 subset_size=4, walk_scale=1.5, traverse_scale=6,
                 move_probabilities=(60/122, 60/122., 1/122., 1/122.), **kwargs):
        """The thoughtful- or traverse- walk MCMC algorithm.

        The t-walk (twalk) algorithm of Christen and Fox [1] is a general-purpose MCMC algorithm that
        requires no tuning, is scale-invariant, is technically non-adaptive (but self-adjusting), and can sample
        from target distributions with arbitrary scale and correlation structures. A random subset of one of two
        vectors is moved around the state-space to influence one of two chains, per iteration.

        Args:
            ll_func (mot.lib.cl_function.CLFunction): The log-likelihood function.
            log_prior_func (mot.lib.cl_function.CLFunction): The log-prior function.
            x0 (ndarray): the starting positions for the sampler. Should be a two dimensional matrix
                with for every modeling instance (first dimension) and every parameter (second dimension) a value.
            x1 (ndarray): the second coordinate vector for every model instance. This should be of the same size as
                ``x0``, but then with different values. This basically defines the initial sampling space.
            subset_size (int): the expected number of parameters to move in one step.
                This is variable n_1 in the article [1]. If there are less parameters than the subset_size, we will
                use all parameters.
            walk_scale (float): the scalar of the walk move. This is parameter a_w in [1]. A reasonable range
                given by [1] is [0.3, 2]
            traverse_scale (float): the scalar of the traverse move. This is parameter a_t in [1]. A reasonable range
                given by [1] is [2, 10]
            move_probabilities (Tuple[float]): the probabilities of the four different moves supported in the
                t-walk algorithm. Defaults to (0.4918, 0.4918, 0.0082, 0.0082) [1], with respectively the probability
                of walk, traverse, hop, blow moves.
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

                    void <func_name>(void* data, mot_float_type* x);

        References:
            [1] Christen JA, Foxy C. A general purpose sampling algorithm for continuous distributions (the t-walk).
                Bayesian Anal. 2010;5(2):263-282. doi:10.1214/10-BA603.
        """
        super().__init__(ll_func, log_prior_func, x0, **kwargs)

        self._subset_size = subset_size
        self._param_choose_prob = min(self._nmr_params, self._subset_size) / (float(self._nmr_params))
        self._walk_scale = walk_scale
        self._traverse_scale = traverse_scale
        self._move_probabilities = move_probabilities

        float_type = self._cl_runtime_info.mot_float_dtype
        if x1.shape != x0.shape:
            self._x1 = np.require(np.tile(x1, (x0.shape[0], 1)), dtype=float_type)
        else:
            self._x1 = np.require(np.copy(x1), dtype=float_type)

        self._x1_log_likelihood = np.zeros(self._nmr_problems, dtype=float_type)
        self._x1_log_prior = np.zeros(self._nmr_problems, dtype=float_type)

        self._finalize_proposal_func = finalize_proposal_func or SimpleCLFunction.from_string(
            'void finalizeProposal(void* data, mot_float_type* x){}')

        self._initialize_likelihood_prior(self._x1, self._x1_log_likelihood, self._x1_log_prior)

    def _get_mcmc_method_kernel_data(self):
        return Struct({
            'x1_position': Array(self._x1, 'mot_float_type', mode='rw'),
            'x1_log_likelihood': Array(self._x1_log_likelihood, 'mot_float_type', mode='rw'),
            'x1_log_prior': Array(self._x1_log_prior, 'mot_float_type', mode='rw'),
            'scratch_mft': LocalMemory('mot_float_type', self._nmr_params + 2),
            'scratch_int': LocalMemory('int', self._nmr_params + 4),
        }, '_twalk_data')

    def _get_state_update_cl_func(self, nmr_samples, thinning, return_output):
        func = parse_cl_function('''
            void _twalk_advance_chain(
                    void* method_data,
                    void* data,
                    ulong current_iteration,
                    int chain_ind,
                    mot_float_type* current_position,
                    mot_float_type* current_log_likelihood,
                    mot_float_type* current_log_prior){
                _twalk_data* twalk_data = (_twalk_data*)method_data;
                bool is_first_work_item = get_local_id(0) == 0;

                int* kernel_ind = twalk_data->scratch_int + 1;
                int* nmr_params_selected = twalk_data->scratch_int + 2;
                int* proposal_accepted = twalk_data->scratch_int + 3;
                int* params_selector = twalk_data->scratch_int + 4;

                mot_float_type* proposal_ll = twalk_data->scratch_mft;
                mot_float_type* proposal_lprior = twalk_data->scratch_mft + 1;
                mot_float_type* proposal = twalk_data->scratch_mft + 2;
                *proposal_accepted = false;
                *proposal_ll = 0;
                *proposal_lprior = 0;

                mot_float_type* main_chain;
                mot_float_type* helper_chain;
                mot_float_type* main_ll;
                mot_float_type* main_lprior;
                if(is_first_work_item){
                    float r = frand();
                    if(r < ''' + str(self._move_probabilities[0]) + '''){
                        *kernel_ind = 0;
                    }
                    else if(r < ''' + str(sum(self._move_probabilities[:2])) + '''){
                        *kernel_ind = 1;
                    }
                    else if(r < ''' + str(sum(self._move_probabilities[:3])) + '''){
                        *kernel_ind = 2;
                    }
                    else{
                        *kernel_ind = 3;
                    }

                    *nmr_params_selected = 0;
                    for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        params_selector[i] = frand() < ''' + str(self._param_choose_prob) + ''';
                        (*nmr_params_selected)++;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if(chain_ind == 0){
                    main_chain = twalk_data->x1_position;
                    helper_chain = current_position;
                    main_ll = twalk_data->x1_log_likelihood;
                    main_lprior = twalk_data->x1_log_prior;
                }
                else{
                    main_chain = current_position;
                    helper_chain = twalk_data->x1_position;
                    main_ll = current_log_likelihood;
                    main_lprior = current_log_prior;
                }

                if(*kernel_ind == 0){
                    _twalk_walk_move(main_chain, helper_chain, main_ll, main_lprior,
                                     proposal, proposal_ll, proposal_lprior, proposal_accepted,
                                     params_selector, *nmr_params_selected, data);
                }
                else if(*kernel_ind == 1){
                    _twalk_traverse_move(main_chain, helper_chain, main_ll, main_lprior,
                                         proposal, proposal_ll, proposal_lprior, proposal_accepted,
                                         params_selector, *nmr_params_selected, data);
                }
                else if(*kernel_ind == 2){
                    _twalk_hop_move(main_chain, helper_chain, main_ll, main_lprior,
                                    proposal, proposal_ll, proposal_lprior, proposal_accepted,
                                    params_selector, *nmr_params_selected, data);
                }
                else{
                    _twalk_blow_move(main_chain, helper_chain, main_ll, main_lprior,
                                     proposal, proposal_ll, proposal_lprior, proposal_accepted,
                                     params_selector, *nmr_params_selected, data);
                }

                if(is_first_work_item && *proposal_accepted){
                    if(chain_ind == 0){
                        for(uint k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                            twalk_data->x1_position[k] = proposal[k];
                        }
                        *twalk_data->x1_log_likelihood = *proposal_ll;
                        *twalk_data->x1_log_prior = *proposal_lprior;
                    }
                    else{
                        for(uint k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                            current_position[k] = proposal[k];
                        }
                        *current_log_likelihood = *proposal_ll;
                        *current_log_prior = *proposal_lprior;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            void _advanceSampler(
                    void* method_data,
                    void* data,
                    ulong current_iteration,
                    mot_float_type* current_position,
                    mot_float_type* current_log_likelihood,
                    mot_float_type* current_log_prior){
                _twalk_data* twalk_data = (_twalk_data*)method_data;

                int* chain_ind = twalk_data->scratch_int;
                *chain_ind = 0;

                while(*chain_ind != 1){
                    if(get_local_id(0) == 0){
                        *chain_ind = frand() > 0.5;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    _twalk_advance_chain(method_data, data, current_iteration, *chain_ind,
                                         current_position, current_log_likelihood, current_log_prior);
                }
            }

        ''', dependencies=[self._finalize_proposal_func, self._get_walk_move(), self._get_traverse_move(),
                           self._get_hop_move(), self._get_blow_move()])
        return func.get_cl_code()

    def _get_walk_move(self):
        return parse_cl_function('''
            void _twalk_walk_move_proposal(
                    mot_float_type* main_chain,
                    mot_float_type* helper_chain,
                    mot_float_type* proposal,
                    int* params_selector){

                float u, a;
                const float aw = ''' + str(self._walk_scale) + ''';

                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    if(params_selector[i]){
                        u = frand();
                        a = (aw / (1 + aw)) * (-1 + 2 * u + aw * u * u);

                        proposal[i] = main_chain[i] + a * (main_chain[i] - helper_chain[i]);
                    }
                    else{
                        proposal[i] = main_chain[i];
                    }
                }
            }

            void _twalk_walk_move(
                    mot_float_type* main_chain,
                    mot_float_type* helper_chain,
                    mot_float_type* main_ll,
                    mot_float_type* main_lprior,
                    mot_float_type* proposal,
                    mot_float_type* proposal_ll,
                    mot_float_type* proposal_lprior,
                    int* proposal_accepted,
                    int* params_selector,
                    int nmr_params_selected,
                    void* data){

                bool is_first_work_item = get_local_id(0) == 0;

                if(is_first_work_item){
                    _twalk_walk_move_proposal(main_chain, helper_chain, proposal, params_selector);
                    ''' + self._finalize_proposal_func.get_cl_function_name() + '''(data, proposal);

                    *proposal_lprior = _computeLogPrior(proposal, data);
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if(exp(*proposal_lprior) > 0){
                    *proposal_ll = _computeLogLikelihood(proposal, data);

                    if(is_first_work_item){
                        *proposal_accepted = frand() < exp((*proposal_ll + *proposal_lprior)
                                                           - (*main_ll + *main_lprior));
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        ''')

    def _get_traverse_move(self):
        return parse_cl_function('''
            float _twalk_traverse_move_compute_beta(){
                float4 r = frand4();
                float at = ''' + str(self._traverse_scale) + ''';

                if(r.x < (at - 1.0) / (2.0 * at)){
                    return exp(1.0 / (at + 1.0) * log(r.y));
                }
                return exp(1.0 / (1.0 - at) * log(r.y));
            }

            void _twalk_traverse_move_proposal(
                    mot_float_type* main_chain,
                    mot_float_type* helper_chain,
                    mot_float_type* proposal,
                    int* params_selector,
                    mot_float_type beta){
                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    if(params_selector[i]){
                        proposal[i] = helper_chain[i] + beta * (helper_chain[i] - main_chain[i]);
                    }
                    else{
                        proposal[i] = main_chain[i];
                    }
                }
            }

            void _twalk_traverse_move(
                    mot_float_type* main_chain,
                    mot_float_type* helper_chain,
                    mot_float_type* main_ll,
                    mot_float_type* main_lprior,
                    mot_float_type* proposal,
                    mot_float_type* proposal_ll,
                    mot_float_type* proposal_lprior,
                    int* proposal_accepted,
                    int* params_selector,
                    int nmr_params_selected,
                    void* data){
                float beta;
                bool is_first_work_item = get_local_id(0) == 0;

                if(is_first_work_item){
                    beta = _twalk_traverse_move_compute_beta();
                    _twalk_traverse_move_proposal(main_chain, helper_chain, proposal,
                                                  params_selector, beta);
                    ''' + self._finalize_proposal_func.get_cl_function_name() + '''(data, proposal);

                    *proposal_lprior = _computeLogPrior(proposal, data);
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if(exp(*proposal_lprior) > 0){
                    *proposal_ll = _computeLogLikelihood(proposal, data);

                    if(is_first_work_item){
                        if(nmr_params_selected == 0){
                            *proposal_accepted = true;
                        }
                        else{
                            *proposal_accepted = frand() < exp((*proposal_ll + *proposal_lprior)
                                                               - (*main_ll + *main_lprior)
                                                               + (nmr_params_selected - 2) * log(beta));
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        ''')

    def _get_hop_move(self):
        return parse_cl_function('''
            void _twalk_hop_move_proposal(
                    mot_float_type* main_chain,
                    mot_float_type* helper_chain,
                    mot_float_type* proposal,
                    int* params_selector){
                mot_float_type sigma = 0;
                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    sigma = max(sigma, params_selector[i] * fabs(main_chain[i] - helper_chain[i]));
                }
                sigma /= 3.0;

                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    if(params_selector[i]){
                        proposal[i] = main_chain[i] + sigma * frandn();
                    }
                    else{
                        proposal[i] = main_chain[i];
                    }
                }
            }

            float _hop_move_hasting_criteria(
                    mot_float_type* proposal,
                    mot_float_type* main_chain,
                    mot_float_type* helper_chain,
                    int* params_selector,
                    int nmr_params_selected){

                mot_float_type sigma = 0;
                double sum = 0;
                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    sigma = max(sigma, params_selector[i] * fabs(main_chain[i] - helper_chain[i]));
                    sum += pown(proposal[i] - main_chain[i], 2);
                }
                sigma /= 3.0;

                if(nmr_params_selected > 0){
                    return -(nmr_params_selected/2.0) * log(2*M_PI)
                           + nmr_params_selected * log(3.0)
                           - nmr_params_selected * log(sigma)
                           - 9 * sum / (2 * sigma * sigma);
                }
                return 0;
            }

            void _twalk_hop_move(
                    mot_float_type* main_chain,
                    mot_float_type* helper_chain,
                    mot_float_type* main_ll,
                    mot_float_type* main_lprior,
                    mot_float_type* proposal,
                    mot_float_type* proposal_ll,
                    mot_float_type* proposal_lprior,
                    int* proposal_accepted,
                    int* params_selector,
                    int nmr_params_selected,
                    void* data){
                bool is_first_work_item = get_local_id(0) == 0;

                if(is_first_work_item){
                    _twalk_hop_move_proposal(main_chain, helper_chain, proposal, params_selector);
                    ''' + self._finalize_proposal_func.get_cl_function_name() + '''(data, proposal);

                    *proposal_lprior = _computeLogPrior(proposal, data);
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if(exp(*proposal_lprior) > 0){
                    *proposal_ll = _computeLogLikelihood(proposal, data);

                    if(is_first_work_item){
                        float g_xy = _hop_move_hasting_criteria(main_chain, proposal, helper_chain,
                                                                params_selector, nmr_params_selected);
                        float g_yx = _hop_move_hasting_criteria(proposal, main_chain, helper_chain,
                                                                params_selector, nmr_params_selected);

                        *proposal_accepted = frand() < exp((*proposal_ll + *proposal_lprior)
                                                           - (*main_ll + *main_lprior)
                                                           + (g_xy - g_yx));
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        ''')

    def _get_blow_move(self):
        return parse_cl_function('''
            void _twalk_blow_move_proposal(
                    mot_float_type* main_chain,
                    mot_float_type* helper_chain,
                    mot_float_type* proposal,
                    int* params_selector){

                mot_float_type sigma = 0;
                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    sigma = max(sigma, params_selector[i] * fabs(main_chain[i] - helper_chain[i]));
                }

                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    if(params_selector[i]){
                        proposal[i] = helper_chain[i] + sigma * frandn();
                    }
                    else{
                        proposal[i] = main_chain[i];
                    }
                }
            }

            float _blow_move_hasting_criteria(
                    mot_float_type* proposal,
                    mot_float_type* main_chain,
                    mot_float_type* helper_chain,
                    int* params_selector,
                    int nmr_params_selected){

                mot_float_type sigma = 0;
                double sum = 0;
                for(uint i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                    sigma = max(sigma, params_selector[i] * fabs(main_chain[i] - helper_chain[i]));
                    sum += pown(proposal[i] - helper_chain[i], 2);
                }

                if(nmr_params_selected > 0){
                    return -(nmr_params_selected/2.0) * log(2*M_PI)
                           - nmr_params_selected * log(sigma)
                           - sum / (2 * sigma * sigma);
                }
                return 0;
            }

            void _twalk_blow_move(
                    mot_float_type* main_chain,
                    mot_float_type* helper_chain,
                    mot_float_type* main_ll,
                    mot_float_type* main_lprior,
                    mot_float_type* proposal,
                    mot_float_type* proposal_ll,
                    mot_float_type* proposal_lprior,
                    int* proposal_accepted,
                    int* params_selector,
                    int nmr_params_selected,
                    void* data){

                bool is_first_work_item = get_local_id(0) == 0;

                if(is_first_work_item){
                    _twalk_blow_move_proposal(main_chain, helper_chain, proposal, params_selector);
                    ''' + self._finalize_proposal_func.get_cl_function_name() + '''(data, proposal);

                    *proposal_lprior = _computeLogPrior(proposal, data);
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if(exp(*proposal_lprior) > 0){
                    *proposal_ll = _computeLogLikelihood(proposal, data);

                    if(is_first_work_item){
                        float g_xy = _blow_move_hasting_criteria(main_chain, proposal, helper_chain,
                                                                 params_selector, nmr_params_selected);
                        float g_yx = _blow_move_hasting_criteria(proposal, main_chain, helper_chain,
                                                                 params_selector, nmr_params_selected);

                        *proposal_accepted = frand() < exp((*proposal_ll + *proposal_lprior)
                                                           - (*main_ll + *main_lprior)
                                                           + (g_xy - g_yx));
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        ''')
