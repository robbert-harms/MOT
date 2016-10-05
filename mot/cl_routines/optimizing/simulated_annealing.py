from mot.utils import get_ranlux_cl
from .base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SimulatedAnnealing(AbstractParallelOptimizer):

    default_patience = 500

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=False, patience=None,
                 optimizer_options=None, **kwargs):
        """Use Simulated Annealing to calculate the optimum.

        This does not use the parameter codec, even if set to True. This because the priors (should) already
        cover the bounds.

        This implementation uses an adapted Metropolis Hasting algorithm to find the optimum. This being a sampling
        routine it does require that the model is a implementation of SampleModelInterface.

        Args:
            patience (int):
                Used to set the maximum number of samples to patience*(number_of_parameters+1)
            optimizer_options (dict): the optimization options. Contains:
                - proposal_update_intervals (int): the interval by which we update the proposal std.

        """
        patience = patience or self.default_patience
        super(SimulatedAnnealing, self).__init__(cl_environments, load_balancer, False, patience=patience, **kwargs)
        optimizer_options = optimizer_options or {}
        self.proposal_update_intervals = optimizer_options.get('proposal_update_intervals', 50)
        self._annealing_schedule = ExponentialCoolingSchedule()
        self._initial_temperature_strategy = SimpleInitialTemperatureStrategy()

    def _get_worker_generator(self, *args):
        return lambda cl_environment: SimulatedAnnealingWorker(cl_environment, *args, patience=self.patience,
                                                               proposal_update_intervals=self.proposal_update_intervals,
                                                               annealing_schedule=self._annealing_schedule,
                                                               init_temp_strategy=self._initial_temperature_strategy)


class SimulatedAnnealingWorker(AbstractParallelOptimizerWorker):

    def __init__(self, *args, **kwargs):
        self.patience = kwargs.pop('patience')
        self.proposal_update_intervals = kwargs.pop('proposal_update_intervals')
        self._annealing_schedule = kwargs.pop('annealing_schedule')
        self._init_temp_strategy = kwargs.pop('init_temp_strategy')

        super(SimulatedAnnealingWorker, self).__init__(*args, **kwargs)
        self._use_param_codec = False

    def _get_optimizer_cl_code(self):
        kernel_source = self._get_evaluate_function()

        kernel_source += '#define RANLUXCL_LUX 4' + "\n"
        kernel_source += get_ranlux_cl()

        kernel_source += self._model.get_log_prior_function('getLogPrior')
        kernel_source += self._model.get_proposal_function('getProposal')
        kernel_source += self._model.get_proposal_state_update_function('updateProposalState')

        if not self._model.is_proposal_symmetric():
            kernel_source += self._model.get_proposal_logpdf('getProposalLogPDF')

        kernel_source += self._model.get_log_likelihood_function('getLogLikelihood', full_likelihood=False)

        kernel_source += self._get_sampling_code()
        return kernel_source

    def _get_optimizer_call_name(self):
        return 'simulated_annealing'

    def _get_sampling_code(self):
        proposal_state_size = len(self._model.get_proposal_state())
        proposal_state = '{' + ', '.join(map(str, self._model.get_proposal_state())) + '}'
        acceptance_counters_between_proposal_updates = '{' + ', '.join('0' * self._nmr_params) + '}'

        kernel_source = self._annealing_schedule.get_temperature_update_cl_function()
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
                               void* rand_settings,
                               double* const current_likelihood,
                               mot_float_type* const current_prior,
                               const optimize_data* const data,
                               mot_float_type* const proposal_state,
                               uint* const ac_between_proposal_updates,
                               mot_float_type* temperature){

                mot_float_type new_prior;
                double new_likelihood;
                double bayesian_f;
                mot_float_type old_x;

                #pragma unroll 1
                for(int k = 0; k < ''' + str(self._nmr_params) + '''; k++){

                    old_x = x[k];
                    x[k] = getProposal(k, x[k], (ranluxcl_state_t*)rand_settings, proposal_state);

                    new_prior = getLogPrior(x);

                    if(exp(new_prior) > 0){
                        new_likelihood = getLogLikelihood(data, x);
        '''
        if self._model.is_proposal_symmetric():
            kernel_source += '''
                        bayesian_f = exp(((new_likelihood + new_prior) -
                                            (*current_likelihood + *current_prior)) / *temperature);
                '''
        else:
            kernel_source += '''
                        mot_float_type x_to_prop = getProposalLogPDF(k, old_x, x[k], proposal_state);
                        mot_float_type prop_to_x = getProposalLogPDF(k, x[k], x[k], proposal_state);

                        bayesian_f = exp(((new_likelihood + new_prior + x_to_prop) -
                                          (*current_likelihood + *current_prior + prop_to_x)) / *temperature);
                '''
        kernel_source += '''
                        if(new_likelihood > *current_likelihood || rand(rand_settings) < bayesian_f){
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
        '''
        kernel_source += self._init_temp_strategy.get_init_temp_cl_function()
        kernel_source += '''
            int simulated_annealing(mot_float_type* const x, const void* const data, void* rand_settings){
                uint proposal_update_count = 0;

                mot_float_type proposal_state[] = ''' + proposal_state + ''';
                uint ac_between_proposal_updates[] = ''' + acceptance_counters_between_proposal_updates + ''';

                double current_likelihood = getLogLikelihood((optimize_data*)data, x);
                mot_float_type current_prior = getLogPrior(x);

                double prev_likelihood = current_likelihood;
                mot_float_type prev_prior = current_prior;

                rand(rand_settings);

                mot_float_type temperature;
                mot_float_type min_temp;
                mot_float_type initial_temp;
                its_get_initial_temperature(&temperature, &min_temp, &initial_temp,
                                            x, rand_settings, &current_likelihood, &current_prior,
                                            data, proposal_state, ac_between_proposal_updates,
                                            &proposal_update_count);

                for(uint step = 0; step < ''' + str(self.patience * (self._nmr_params + 1)) + '''; step++){
                    as_update_temperature(&temperature, min_temp, initial_temp, step,
                                          (uint) ''' + str(self.patience * (self._nmr_params + 1)) + ''');

                    if(temperature <= 0.0f){
                        return 2;
                    }

                    _update_state(x, rand_settings, &current_likelihood, &current_prior,
                                  data, proposal_state, ac_between_proposal_updates, &temperature);
                    _update_proposals(proposal_state, ac_between_proposal_updates, &proposal_update_count);
                }

                return 1;
            }
        '''
        return kernel_source

    def _uses_random_numbers(self):
        return True


class InitialTemperatureStrategy(object):

    def __init__(self, *args, **kwargs):
        """The base class for strategies for determining the initial temperature to use during annealing.
        """

    def get_init_temp_cl_function(self):
        """Get the function called by the annealing routine to determine the initial temperature.
        """


class SimpleInitialTemperatureStrategy(InitialTemperatureStrategy):

    def __init__(self, min_temp=0, initial_temp=1e5, **kwargs):
        """Sets the initial temperature to a predefined value.
        """
        super(SimpleInitialTemperatureStrategy, self).__init__(**kwargs)
        self.min_temp = min_temp
        self.initial_temp = initial_temp

    def get_init_temp_cl_function(self):
        return '''
            void its_get_initial_temperature(
                    mot_float_type* temperature,
                    mot_float_type* min_temp,
                    mot_float_type* initial_temp,
                    mot_float_type* const x,
                    void* rand_settings,
                    double* const current_likelihood,
                    mot_float_type* const current_prior,
                    const optimize_data* const data,
                    mot_float_type* const proposal_state,
                    uint* const ac_between_proposal_updates,
                    uint* const proposal_update_count){

                *min_temp = ''' + str(self.min_temp) + ''';
                *initial_temp = ''' + str(self.initial_temp) + ''';
                *temperature = *initial_temp;
            }
        '''


class AnnealingSchedule(object):

    def __init__(self, *args, **kwargs):
        """The base class for the annealing schedule.

        Implementing classes can implement an annealing schedule to update the temperature during annealing.
        """

    def get_temperature_update_cl_function(self):
        """Get the function called by the annealing routine to update the temperature.

        This must return a CL string with a function with the following signature
        (where the prefix 'as' stands for Annealing Schedule):

        .. code-block:: c

            void as_update_temperature(mot_float_type* temperature, const mot_float_type min_temp,
                                       const mot_float_type initial_temp, const uint step, const uint nmr_steps);

        Here ``temperature`` is the value to control, ``min_temp`` is the minimum temperature allowed,
        ``initial_temp`` was the initial temperature, ``step`` is the current step and ``nmr_steps``
        is the maximum number of steps.

        Returns:
            str: the temperature update function
        """


class ExponentialCoolingSchedule(AnnealingSchedule):

    def __init__(self, damping_factor=0.95, **kwargs):
        """A simple exponential cooling schedule.

        The temperature ``T`` at time ``k+1`` is given by:

        .. code-block:: c

            T_k+1 = a * T_k

        Where ``0 < a < 1`` is the damping factor.
        """
        self.damping_factor = damping_factor
        super(ExponentialCoolingSchedule, self).__init__(**kwargs)

    def get_temperature_update_cl_function(self):
        return '''
            void as_update_temperature(mot_float_type* temperature, const mot_float_type min_temp,
                                       const mot_float_type initial_temp, const uint step, const uint nmr_steps){
                *temperature *= ''' + (str(self.damping_factor)) + ''';

                 if(*temperature < min_temp){
                    *temperature = min_temp;
                 }
            }
        '''
