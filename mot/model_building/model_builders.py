from textwrap import dedent, indent

import numpy as np
import copy
from six import string_types
from mot.cl_data_type import SimpleCLDataType
from mot.cl_routines.mapping.codec_runner import CodecRunner
from mot.cl_routines.sampling.metropolis_hastings import DefaultMHState
from mot.model_building.model_function_priors import ModelFunctionPrior
from mot.model_building.model_functions import Weight, ModelFunction
from mot.model_building.parameters import CurrentObservationParam, StaticMapParameter, ProtocolParameter, \
    ModelDataParameter, FreeParameter
from mot.model_building.data_adapter import SimpleDataAdapter
from mot.model_building.parameter_functions.dependencies import SimpleAssignment, AbstractParameterDependency
from mot.model_building.utils import ParameterCodec, SimpleModelPrior
from mot.model_interfaces import OptimizeModelInterface, SampleModelInterface, KernelDataInfo
from mot.utils import is_scalar, all_elements_equal, get_single_value, topological_sort, \
    SimpleNamedCLFunction

__author__ = 'Robbert Harms'
__date__ = "2014-03-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class OptimizeModelBuilder(object):

    def __init__(self, name, model_tree, evaluation_model, signal_noise_model=None,
                 problem_data=None, enforce_weights_sum_to_one=True):
        """Create a new model builder that can construct an optimization model from a combination of model functions.

        Args:
            name (str): the name of the model
            model_tree (mot.model_building.trees.CompartmentModelTree): the model tree object
            evaluation_model (mot.model_building.evaluation_models.EvaluationModel): the evaluation model to
                use for the resulting complete model
            signal_noise_model (mot.model_building.signal_noise_models.SignalNoiseModel): the optional signal
                noise model to use to add noise to the model prediction
            problem_data (ProblemData): the problem data object
            enforce_weights_sum_to_one (boolean): if we want to enforce that weights sum to one. This does the
                following things; it fixes the first weight to the sum of the others and it adds a transformation
                that ensures that those other weights sum to at most one.
        """
        super(OptimizeModelBuilder, self).__init__()
        self._name = name
        self._model_tree = model_tree
        self._evaluation_model = evaluation_model
        self._signal_noise_model = signal_noise_model
        self._kernel_data_struct_type = '_model_data'

        self._enforce_weights_sum_to_one = enforce_weights_sum_to_one

        self._double_precision = False

        self._model_functions_info = self._init_model_information_container(
            model_tree, evaluation_model, signal_noise_model)

        self._lower_bounds = {'{}.{}'.format(m.name, p.name): p.lower_bound for m, p in
                              self._model_functions_info.get_free_parameters_list()}

        self._upper_bounds = {'{}.{}'.format(m.name, p.name): p.upper_bound for m, p in
                              self._model_functions_info.get_free_parameters_list()}

        self._problem_data = None
        if problem_data:
            self.set_problem_data(problem_data)

        self._set_default_dependencies()

    def _init_model_information_container(self, model_tree, evaluation_model, signal_noise_model):
        """Get the model information container object.

        The rationale for this function is that some subclasses may have additional parameters not present in
        optimization. For example, in sampling one can have priors with parameters. These parameters must be
        added to the model and the best point to do that is in the ModelFunctionsInformation object.

        Returns:
            ModelFunctionsInformation: the model function information object
        """
        return ModelFunctionsInformation(model_tree, evaluation_model, signal_noise_model)

    def get_composite_model_function(self):
        """Get the composite model function for the current model tree and possible signal noise model.

        Returns:
            CompositeModelFunction: the model function for the composite model
        """
        return CompositeModelFunction(self._model_tree, signal_noise_model=self._signal_noise_model)

    def build(self, problems_to_analyze=None):
        """Construct the final immutable model with the current settings.

        Args:
            problems_to_analyze (ndarray): construct the model such that it analyzes only a subset of the problems

        Returns:
            OptimizeModelInterface: an implementation an optimization model with all the current settings

        Raises:
            RuntimeError: if some of the required items are not set prior to building.
        """
        if self._problem_data is None:
            raise RuntimeError('Problem data is not set, can not build the model.')

        return SimpleOptimizeModel(problems_to_analyze,
                                   self.name,
                                   self.double_precision,
                                   self.get_free_param_names(),
                                   self._get_kernel_data_info(problems_to_analyze),
                                   self._get_nmr_problems(problems_to_analyze),
                                   self.get_nmr_inst_per_problem(),
                                   self.get_nmr_estimable_parameters(),
                                   self._get_initial_parameters(problems_to_analyze),
                                   self._get_pre_eval_parameter_modifier(),
                                   self._get_model_eval_function(problems_to_analyze),
                                   self._get_observation_return_function(),
                                   self._get_objective_per_observation_function(problems_to_analyze),
                                   self.get_lower_bounds(),
                                   self.get_upper_bounds())

    @property
    def name(self):
        """See super class OptimizeModelInterface for details"""
        return self._name

    @property
    def double_precision(self):
        """See super class OptimizeModelInterface for details"""
        return self._double_precision

    @double_precision.setter
    def double_precision(self, value):
        """Set the value for double_precision.

        Args:
            value (boolean): if we would like to do the computations in double of single floating point type.
        """
        self._double_precision = value

    def fix(self, model_param_name, value):
        """Fix the given model.param to the given value.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'
            value (scalar or vector or string or AbstractParameterDependency): The value or dependency
                to fix the given parameter to.

        Returns:
            Returns self for chainability
        """
        if isinstance(value, string_types):
            value = SimpleAssignment(value)
        self._model_functions_info.fix_parameter(model_param_name, value)
        return self

    def unfix(self, model_param_name):
        """Unfix the given model.param

        Args:
            model_param_name (string): A model.param name like 'Ball.d'

        Returns:
            Returns self for chainability
        """
        self._model_functions_info.unfix(model_param_name)
        return self

    def init(self, model_param_name, value):
        """Init the given model.param to the given value.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'
            value (scalar or vector): The value to initialize the given parameter to

        Returns:
            Returns self for chainability
        """
        if not self._model_functions_info.is_fixed(model_param_name):
            self._model_functions_info.set_parameter_value(model_param_name, value)
        return self

    def set_initial_parameters(self, initial_params):
        """Update the initial parameters for this model by the given values.

        This only affects free parameters.

        Args:
            initial_params (dict): a dictionary containing as keys full parameter names (<model>.<param>) and as values
                numbers or arrays to be used as starting point
        """
        for m, p in self._model_functions_info.get_free_parameters_list():
            param_name = '{}.{}'.format(m.name, p.name)

            if param_name in initial_params:
                self.init(param_name, initial_params[param_name])

        return self

    def set_lower_bound(self, model_param_name, value):
        """Set the lower bound for the given parameter to the given value.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'
            value (scalar or vector): The value to set the lower bounds to

        Returns:
            Returns self for chainability
        """
        self._lower_bounds[model_param_name] = value
        return self

    def set_lower_bounds(self, lower_bounds):
        """Apply multiple lower bounds from a dictionary.

        Args:
            lower_bounds (dict): per parameter a lower bound

        Returns:
            Returns self for chainability
        """
        for param, value in lower_bounds.items():
            self.set_lower_bound(param, value)
        return self

    def set_upper_bound(self, model_param_name, value):
        """Set the upper bound for the given parameter to the given value.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'
            value (scalar or vector): The value to set the upper bounds to

        Returns:
            Returns self for chainability
        """
        self._upper_bounds[model_param_name] = value
        return self

    def set_upper_bounds(self, upper_bounds):
        """Apply multiple upper bounds from a dictionary.

        Args:
            upper_bounds (dict): per parameter a upper bound

        Returns:
            Returns self for chainability
        """
        for param, value in upper_bounds.items():
            self.set_upper_bound(param, value)
        return self

    def has_parameter(self, model_param_name):
        """Check to see if the given parameter is defined in this model.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'

        Returns:
            boolean: true if the parameter is defined in this model, false otherwise.
        """
        return self._model_functions_info.has_parameter(model_param_name)

    def set_problem_data(self, problem_data):
        """Set the problem data this model will deal with.

        This will also call the function set_noise_level_std() with the noise_std from the new problem data.

        Args:
            problem_data (mot.model_building.problem_data.AbstractProblemData):
                The container for the problem data we will use for this model.

        Returns:
            Returns self for chainability
        """
        self._problem_data = problem_data
        if self._problem_data.noise_std is not None:
            self._model_functions_info.set_parameter_value('{}.{}'.format(
                self._evaluation_model.name,
                self._evaluation_model.get_noise_std_param_name()), self._problem_data.noise_std)
        return self

    def get_problem_data(self):
        """Get the problem data actually being used by this model.

        Returns:
            mot.model_building.problem_data.AbstractProblemData: the problem data being used by this model
        """
        return self._problem_data

    def get_required_protocol_names(self):
        """Get a list with the constant data names that are needed for this model to work.

        For example, an implementing diffusion MRI model might require the presence of the protocol parameter
        'g' and 'b'. This function should then return ('g', 'b').

        Returns:
            list: A list of columns names that need to be present in the protocol
        """
        return list(set([p.name for m, p in self._model_functions_info.get_model_parameter_list() if
                         isinstance(p, ProtocolParameter)]))

    def set_fixed_parameter_values(self, fixed_values):
        """Given a dictionary with static maps, initialize the values of the static parameters with these values.

        Make sure that if vectors are given, the lengt of the vector should match the length of the number of problems
        (in the problem data).

        Args:
            fixed_values (dict): the dictionary with the static maps.
        """
        static_params = self._model_functions_info.get_static_parameters_list()
        for m, p in static_params:
            if p.name in fixed_values:
                self._model_functions_info.set_parameter_value('{}.{}'.format(m.name, p.name), fixed_values[p.name])
        return self

    def get_free_param_names(self):
        """See super class for details"""
        return ['{}.{}'.format(m.name, p.name) for m, p in self._model_functions_info.get_estimable_parameters_list()]

    def get_nmr_inst_per_problem(self):
        """See super class for details"""
        return self._problem_data.get_nmr_inst_per_problem()

    def get_nmr_estimable_parameters(self):
        """See super class for details"""
        return len(self._model_functions_info.get_estimable_parameters_list())

    def get_lower_bounds(self):
        """See super class for details"""
        return [self._lower_bounds['{}.{}'.format(m.name, p.name)] for m, p in
                self._model_functions_info.get_estimable_parameters_list()]

    def get_upper_bounds(self):
        """See super class for details"""
        return [self._upper_bounds['{}.{}'.format(m.name, p.name)] for m, p in
                self._model_functions_info.get_estimable_parameters_list()]

    def get_parameter_codec(self):
        """Get a parameter codec that can be used to transform the parameters to and from optimization and model space.

        This is typically used as input to the ParameterTransformedModel decorator model.

        Returns:
            mot.model_building.utils.ParameterCodec: an instance of a parameter codec
        """
        model_builder = self

        class Codec(ParameterCodec):
            def get_parameter_decode_function(self, function_name='decodeParameters'):
                func = '''
                    void ''' + function_name + '''(const void* data_void, mot_float_type* x){
                '''
                func += model_builder._kernel_data_struct_type
                func += '* data = (' + model_builder._kernel_data_struct_type + '*)data_void;'

                for d in model_builder._get_parameter_transformations()[1]:
                    func += "\n" + "\t" * 4 + d.format('x')

                if model_builder._enforce_weights_sum_to_one:
                    func += model_builder._get_weight_sum_to_one_transformation()

                return func + '''
                    }
                '''

            def get_parameter_encode_function(self, function_name='encodeParameters'):
                func = '''
                    void ''' + function_name + '''(const void* data_void, mot_float_type* x){
                '''

                if model_builder._enforce_weights_sum_to_one:
                    func += model_builder._get_weight_sum_to_one_transformation()

                func += model_builder._kernel_data_struct_type
                func += '* data = (' + model_builder._kernel_data_struct_type + '*)data_void;'

                for d in model_builder._get_parameter_transformations()[0]:
                    func += "\n" + "\t" * 4 + d.format('x')

                return func + '''
                    }
                '''
        return Codec()

    def _get_nmr_problems(self, problems_to_analyze):
        """See super class for details"""
        if problems_to_analyze is None:
            if self._problem_data:
                return self._problem_data.get_nmr_problems()
            return 0
        return len(problems_to_analyze)

    def _get_kernel_data_info(self, problems_to_analyze):
        info = self._get_all_kernel_source_items(problems_to_analyze)

        data_struct_init = info['data_struct_init']
        struct_code = '0'
        if data_struct_init:
            struct_code = ', '.join(data_struct_init)

        data = []
        for data_dict in [self._get_variable_data(problems_to_analyze),
                          self._get_protocol_data(),
                          self._get_static_data()]:
            for el in data_dict.values():
                data.append(el.get_opencl_data())

        return SimpleKernelDataInfo(
            data, info['kernel_param_names'], info['data_struct'],
            self._kernel_data_struct_type,
            (self._kernel_data_struct_type + ' {variable_name} = {{' + struct_code + '}};'))

    def _get_initial_parameters(self, problems_to_analyze):
        np_dtype = np.float32
        if self.double_precision:
            np_dtype = np.float64

        starting_points = []
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            param_name = '{}.{}'.format(m.name, p.name)
            value = self._model_functions_info.get_parameter_value(param_name)

            if is_scalar(value):
                if self._get_nmr_problems(problems_to_analyze) == 0:
                    starting_points.append(np.full((1, 1), value, dtype=np_dtype))
                else:
                    starting_points.append(np.full((self._get_nmr_problems(problems_to_analyze), 1), value,
                                                   dtype=np_dtype))
            else:
                if len(value.shape) < 2:
                    value = np.transpose(np.asarray([value]))
                elif value.shape[1] > value.shape[0]:
                    value = np.transpose(value)
                else:
                    value = value

                if problems_to_analyze is None:
                    starting_points.append(value)
                else:
                    starting_points.append(value[problems_to_analyze, ...])

        starting_points = np.concatenate([np.transpose(np.array([s]))
                                          if len(s.shape) < 2 else s for s in starting_points], axis=1)

        data_adapter = SimpleDataAdapter(starting_points, SimpleCLDataType.from_string('mot_float_type'),
                                         self._get_mot_float_type())
        return data_adapter.get_opencl_data()

    def _get_observation_return_function(self):
        func_name = '_getObservation'
        if self.get_nmr_inst_per_problem() < 2:
            func = '''
                double ''' + func_name + '''(const void* const data, const uint observation_index){
                    return ((''' + self._kernel_data_struct_type + '''*)data)->var_data_observations;
                }
            '''
        else:
            func = '''
                double ''' + func_name + '''(const void* const data, const uint observation_index){
                    return ((''' + \
                   self._kernel_data_struct_type + '''*)data)->var_data_observations[observation_index];
                }
            '''
        return SimpleNamedCLFunction(func, func_name)

    def _get_pre_eval_parameter_modifier(self):
        func_name = '_modifyParameters'
        func = '''
            void ''' + func_name + '''(const void* const data, mot_float_type* x){
            }
        '''
        return SimpleNamedCLFunction(func, func_name)

    def _get_model_eval_function(self, problems_to_analyze):
        composite_model_function = self.get_composite_model_function()

        def get_preliminary():
            cl_preliminary = ''
            cl_preliminary += composite_model_function.get_cl_code()
            pre_model_function = self._get_pre_model_expression_eval_function()
            if pre_model_function:
                cl_preliminary += pre_model_function
            return cl_preliminary

        def get_function_body():
            param_listing = self._get_parameters_listing(
                exclude_list=['{}.{}'.format(m.name, p.name).replace('.', '_') for (m, p) in
                              self._model_functions_info.get_non_model_eval_param_listing()])

            body = ''
            body += self._kernel_data_struct_type + '* data = (' + self._kernel_data_struct_type + '*)void_data; \n'
            body += dedent(param_listing.replace('\t', ' '*4))
            body += self._get_pre_model_expression_eval_code() or ''
            body += '\n'
            body += 'return ' + self._composite_model_to_string(composite_model_function, problems_to_analyze) + ';'
            return body

        function_name = '_evaluateModel'

        cl_function = '''
            double {function_name}(
                    const void* const void_data,
                    const mot_float_type* const x,
                    const uint observation_index){{

                {body}
            }}
        '''.format(function_name=function_name, body=indent(get_function_body(), ' '*4*4)[4*4:])
        cl_function = dedent(cl_function.replace('\t', ' '*4))

        return_str = get_preliminary() + cl_function
        return SimpleNamedCLFunction(return_str, function_name)

    def _get_objective_per_observation_function(self, problems_to_analyze):
        eval_function_info = self._get_model_eval_function(problems_to_analyze)
        obs_func = self._get_observation_return_function()

        param_listing = ''
        for p in self._evaluation_model.get_free_parameters():
            param_listing += self._get_param_listing_for_param(self._evaluation_model, p)

        preliminary = ''
        preliminary += self._evaluation_model.get_cl_dependency_code()

        preliminary += eval_function_info.get_function()
        preliminary += obs_func.get_function()
        preliminary += str(self._evaluation_model.get_objective_per_observation_function(
            '_evaluationModel', eval_function_info.get_name(), obs_func.get_name(), param_listing))

        func_name = 'getObjectiveInstanceValue'
        func = str(preliminary) + '''
            double ''' + func_name + '''(const void* const data, mot_float_type* const x,
                                         const uint observation_index){
                return _evaluationModel(data, x, observation_index);
            }
        '''
        return SimpleNamedCLFunction(func, func_name)

    def _get_parameter_transformations(self):
        dec_func_list = []
        enc_func_list = []
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            name = '{}.{}'.format(m.name, p.name)
            parameter = p
            ind = self._model_functions_info.get_parameter_estimable_index(m, p)
            transform = parameter.parameter_transform

            if all_elements_equal(self._lower_bounds[name]):
                lower_bound = str(get_single_value(self._lower_bounds[name]))
            else:
                lower_bound = 'data->var_data_lb_' + name.replace('.', '_')

            if all_elements_equal(self._upper_bounds[name]):
                upper_bound = str(get_single_value(self._upper_bounds[name]))
            else:
                upper_bound = 'data->var_data_ub_' + name.replace('.', '_')

            s = '{0}[' + str(ind) + '] = ' + transform.get_cl_decode().create_assignment(
                '{0}[' + str(ind) + ']', lower_bound, upper_bound) + ';'

            dec_func_list.append(s)

            s = '{0}[' + str(ind) + '] = ' + transform.get_cl_encode().create_assignment(
                '{0}[' + str(ind) + ']', lower_bound, upper_bound) + ';'

            enc_func_list.append(s)

        return tuple(reversed(enc_func_list)), dec_func_list

    def _transform_observations(self, observations):
        """Apply a transformation on the observations before fitting.

        This function is called by get_problems_var_data() just before the observations are handed over to the
        CL routine.

        To implement any behaviour here, you can override this function and add behaviour that changes the observations.

        Args:
            observations (ndarray): the 2d matrix with the observations. This is the list of
                observations used to build the model (that is, *after* the list has been optionally
                limited with problems_to_analyze).

        Returns:
            observations (ndarray): a 2d matrix of the same shape as the input. This should hold the transformed data.
        """
        return observations

    def _composite_model_to_string(self, composite_model, problems_to_analyze):
        """Create the parameter call code for the composite model.

        Args:
            composite_model (CompositeModelFunction): the composite model function
            problems_to_analyze (list): the problems we are analyzing in this round
        """
        param_list = []
        for model, param in composite_model.get_original_model_parameter_list():
            if isinstance(param, ProtocolParameter):
                param_list.append(param.name)

            elif isinstance(param, ModelDataParameter):
                value = self._model_functions_info.get_parameter_value('{}.{}'.format(model.name, param.name))
                if all_elements_equal(value):
                    param_list.append(str(get_single_value(value)))
                else:
                    param_list.append('data->model_data_' + param.name)

            elif isinstance(param, StaticMapParameter):
                static_map_value = self._get_static_map_value(model, param, problems_to_analyze)
                if all_elements_equal(static_map_value):
                    param_list.append(str(get_single_value(static_map_value)))
                else:
                    if len(static_map_value.shape) > 1 and static_map_value.shape[1] != 1 \
                            and static_map_value.shape[1] == self.get_nmr_inst_per_problem():
                        param_list.append('data->var_data_' + '{}.{}'.format(model.name, param.name).replace('.', '_')
                                          + '[observation_index]')
                    else:
                        param_list.append('data->var_data_' + '{}.{}'.format(model.name, param.name).replace('.', '_'))

            elif isinstance(param, CurrentObservationParam):
                param_list.append('data->var_data_observations[observation_index]')

            else:
                param_list.append('{}.{}'.format(model.name, param.name).replace('.', '_'))

        return composite_model.cl_function_name + '(' + ', '.join(param_list) + ')'

    def _get_parameters_listing(self, exclude_list=()):
        """Get the CL code for the parameter listing, this goes on top of the evaluate function.

        Args:
            exclude_list: an optional list containing parameters to exclude from the listing.
             This should contain full parameter names like: <model_name>_<param_name>

        Returns:
            An CL string that contains all the parameters as primitive data types.
        """
        func = ''
        func += self._get_protocol_parameters_listing(exclude_list=exclude_list)
        func += self._get_fixed_parameters_listing(exclude_list=exclude_list)
        func += self._get_estimable_parameters_listing(exclude_list=exclude_list)
        func += self._get_dependent_parameters_listing(exclude_list=exclude_list)
        return str(func)

    def _get_estimable_parameters_listing(self, exclude_list=()):
        """Get the parameter listing for the free parameters.

        Args:
            exclude_list: a list of parameters to exclude from this listing
        """
        param_list = self._model_functions_info.get_estimable_parameters_list(exclude_priors=True)

        func = ''
        estimable_param_counter = 0
        for m, p in param_list:
            name = '{}.{}'.format(m.name, p.name).replace('.', '_')
            if name not in exclude_list:
                data_type = p.data_type.cl_type
                assignment = 'x[' + str(estimable_param_counter) + ']'
                func += "\t"*4 + data_type + ' ' + name + ' = ' + assignment + ';' + "\n"
                estimable_param_counter += 1
        return func

    def _get_protocol_parameters_listing(self, exclude_list=()):
        """Get the parameter listing for the protocol parameters.

        Args:
            exclude_list: a list of parameters to exclude from this listing
        """
        protocol_info = self._problem_data.protocol
        param_list = self._model_functions_info.get_protocol_parameters_list()

        const_params_seen = []
        func = ''
        for m, p in param_list:
            if ('{}.{}'.format(m.name, p.name).replace('.', '_')) not in exclude_list:
                data_type = p.data_type.cl_type
                if p.name not in const_params_seen:
                    if all_elements_equal(protocol_info[p.name]):
                        if p.data_type.is_vector_type:
                            vector_length = p.data_type.vector_length
                            values = [str(val) for val in protocol_info[p.name][0]]
                            if len(values) < vector_length:
                                values.append(str(0))
                            assignment = '(' + data_type + ')(' + ', '.join(values) + ')'
                        else:
                            assignment = str(float(protocol_info[p.name][0]))
                    else:
                        if p.data_type.is_pointer_type:
                            # this requires generic address spaces available in OpenCL >= 2.0.
                            assignment = '&data->protocol_data_' + p.name + '[observation_index]'
                        else:
                            assignment = 'data->protocol_data_' + p.name + '[observation_index]'
                    func += "\t"*4 + data_type + ' ' + p.name + ' = ' + assignment + ';' + "\n"
                    const_params_seen.append(p.name)
        return func

    def _get_fixed_parameters_listing(self, exclude_list=()):
        """Get the parameter listing for the fixed parameters.

        Args:
            exclude_list: a list of parameters to exclude from this listing
        """
        param_list = self._model_functions_info.get_value_fixed_parameters_list(exclude_priors=True)

        func = ''
        for m, p in param_list:
            name = '{}.{}'.format(m.name, p.name).replace('.', '_')
            if name not in exclude_list:
                data_type = p.data_type.raw_data_type
                value = self._model_functions_info.get_parameter_value('{}.{}'.format(m.name, p.name))

                if all_elements_equal(value):
                    assignment = '(' + data_type + ')' + str(float(get_single_value(value)))
                else:
                    assignment = '(' + data_type + ') data->var_data_' + \
                                 '{}.{}'.format(m.name, p.name).replace('.', '_')
                func += "\t"*4 + data_type + ' ' + name + ' = ' + assignment + ';' + "\n"
        return func

    def _get_dependent_parameters_listing(self, dependent_param_list=None, exclude_list=()):
        """Get the parameter listing for the dependent parameters.

        Args:
            dependent_param_list: the list list of dependent params
            exclude_list: a list of parameters to exclude from this listing, note that this will only exclude the
                definition of the parameter, not the dependency code.
        """
        if dependent_param_list is None:
            dependent_param_list = self._model_functions_info.get_dependency_fixed_parameters_list(exclude_priors=True)

        func = ''
        for m, p in dependent_param_list:
            dependency = self._model_functions_info.get_parameter_value('{}.{}'.format(m.name, p.name))

            if dependency.pre_transform_code:
                func += "\t"*4 + self._convert_parameters_dot_to_bar(dependency.pre_transform_code)

            assignment = self._convert_parameters_dot_to_bar(dependency.assignment_code)
            name = '{}.{}'.format(m.name, p.name).replace('.', '_')
            data_type = p.data_type.raw_data_type

            if ('{}.{}'.format(m.name, p.name).replace('.', '_')) not in exclude_list:
                func += "\t"*4 + data_type + ' ' + name + ' = ' + assignment + ';' + "\n"
        return func

    def _get_fixed_parameters_as_var_data(self, problems_to_analyze):
        var_data_dict = {}
        for m, p in self._model_functions_info.get_value_fixed_parameters_list():
            value = self._model_functions_info.get_parameter_value('{}.{}'.format(m.name, p.name))

            if not all_elements_equal(value):
                if problems_to_analyze is not None:
                    value = value[problems_to_analyze, ...]

                var_data_dict['{}.{}'.format(m.name, p.name).replace('.', '_')] = SimpleDataAdapter(
                    value, p.data_type, self._get_mot_float_type())
        return var_data_dict

    def _get_static_parameters_as_var_data(self, problems_to_analyze):
        static_data_dict = {}

        for m, p in self._model_functions_info.get_static_parameters_list():
            static_map_value = self._get_static_map_value(m, p, problems_to_analyze)

            if not all_elements_equal(static_map_value):
                data_adapter = SimpleDataAdapter(static_map_value, p.data_type, self._get_mot_float_type())
                static_data_dict.update({'{}.{}'.format(m.name, p.name).replace('.', '_'): data_adapter})

        return static_data_dict

    def _get_bounds_as_var_data(self):
        bounds_dict = {}

        for m, p in self._model_functions_info.get_free_parameters_list():
            lower_bound = self._lower_bounds['{}.{}'.format(m.name, p.name)]
            upper_bound = self._upper_bounds['{}.{}'.format(m.name, p.name)]

            if not all_elements_equal(lower_bound):
                data_adapter = SimpleDataAdapter(lower_bound, p.data_type, self._get_mot_float_type())
                bounds_dict.update({'lb_' + '{}.{}'.format(m.name, p.name).replace('.', '_'): data_adapter})

            if not all_elements_equal(upper_bound):
                data_adapter = SimpleDataAdapter(upper_bound, p.data_type, self._get_mot_float_type())
                bounds_dict.update({'ub_' + '{}.{}'.format(m.name, p.name).replace('.', '_'): data_adapter})

        return bounds_dict

    def _get_static_map_value(self, model, parameter, problems_to_analyze):
        """Get the map value for the given parameter of the given model.

        This first checks if the parameter is defined in the static maps data in the problem data. If not, we try
        to get it from the value stored in the parameter itself. If that fails as well we raise an error.

        Also, this only returns the problems for which problems_to_analyze is set.

        Args:
            model (ModelFunction): the model function
            parameter (CLParameter): the parameter for which we want to get the value
            problems_to_analyze (ndarray): the problems we are interested in

        Returns:
            ndarray or number: the value for the given parameter.
        """
        data = None
        value = self._model_functions_info.get_parameter_value('{}.{}'.format(model.name, parameter.name))
        if parameter.name in self._problem_data.static_maps:
            data = self._problem_data.static_maps[parameter.name]
        elif value is not None:
            data = value

        if data is None:
            raise ValueError('No suitable data could be found for the static parameter {}.'.format(parameter.name))

        if is_scalar(data):
            return data

        if problems_to_analyze is not None:
            return data[problems_to_analyze, ...]
        return data

    def _is_non_model_tree_model(self, model):
        return model is self._evaluation_model or (self._signal_noise_model is not None and
                                                   model is self._signal_noise_model)

    def _get_param_listing_for_param(self, m, p):
        """Get the param listing for one specific parameter. This can be used for example for the noise model params.

        Please note, that on the moment this function does not support the complete dependency graph for the dependent
        parameters.
        """
        data_type = p.data_type.raw_data_type
        name = '{}.{}'.format(m.name, p.name).replace('.', '_')
        assignment = ''

        if isinstance(p, ProtocolParameter):
            assignment = 'data->protocol_data_' + p.name + '[observation_index]'
        elif isinstance(p, FreeParameter):
            assignment = self._get_free_parameter_assignment_value(m, p)

        return data_type + ' ' + name + ' = ' + assignment + ';' + "\n"

    def _get_free_parameter_assignment_value(self, m, p):
        """Get the assignment value for one of the free parameters.

        Since the free parameters can be fixed we need an auxiliary routine to get the assignment value.

        Args:
            m: model
            p: parameter
        """
        data_type = p.data_type.raw_data_type
        value = self._model_functions_info.get_parameter_value('{}.{}'.format(m.name, p.name))

        assignment = ''

        if self._model_functions_info.is_fixed_to_value('{}.{}'.format(m.name, p.name)):
            if all_elements_equal(value):
                assignment = '(' + data_type + ')' + str(float(get_single_value(value)))
            else:
                assignment = '(' + data_type + ') data->var_data_{}.{}'.format(m.name, p.name).replace('.', '_')
        elif self._model_functions_info.is_fixed_to_dependency(m, p):
            return self._get_dependent_parameters_listing(((m, p),))
        else:
            ind = self._model_functions_info.get_parameter_estimable_index(m, p)
            assignment += 'x[' + str(ind) + ']'

        return assignment

    def _convert_parameters_dot_to_bar(self, string):
        """Convert a string containing parameters with . to parameter names with _"""
        for m, p in self._model_functions_info.get_model_parameter_list():
            dname = '{}.{}'.format(m.name, p.name)
            bname = '{}.{}'.format(m.name, p.name).replace('.', '_')
            string = string.replace(dname, bname)
        return string

    def _init_fixed_duplicates_dependencies(self):
        """Find duplicate fixed parameters, and make dependencies of them. This saves data transfer in CL."""
        var_data_dict = {}
        for m, p in self._model_functions_info.get_free_parameters_list():
            param_name = '{}.{}'.format(m.name, p.name)
            if self._model_functions_info.is_fixed_to_value(param_name):
                value = self._model_functions_info.get_parameter_value(param_name)

                if not is_scalar(value):
                    duplicate_found = False
                    duplicate_key = None

                    for key, data in var_data_dict.items():
                        if np.array_equal(data, value):
                            duplicate_found = True
                            duplicate_key = key
                            break

                    if duplicate_found:
                        self.fix(param_name, SimpleAssignment(duplicate_key))
                    else:
                        var_data_dict.update({param_name: value})

    def _get_variable_data(self, problems_to_analyze):
        """See super class OptimizeModelInterface for details

        When overriding this function, please note that it should adhere to the attribute problems_to_analyze.
        """
        var_data_dict = {}

        observations = self._problem_data.observations
        if observations is not None:
            if problems_to_analyze is not None:
                observations = observations[problems_to_analyze, ...]

            observations = self._transform_observations(observations)

            data_adapter = SimpleDataAdapter(observations, SimpleCLDataType.from_string('mot_float_type*'),
                                             self._get_mot_float_type())
            var_data_dict.update({'observations': data_adapter})

        var_data_dict.update(self._get_fixed_parameters_as_var_data(problems_to_analyze))
        var_data_dict.update(self._get_static_parameters_as_var_data(problems_to_analyze))
        var_data_dict.update(self._get_bounds_as_var_data())

        return var_data_dict

    def _get_protocol_data(self):
        protocol_info = self._problem_data.protocol
        return_data = {}
        for m, p in self._model_functions_info.get_model_parameter_list():
            if isinstance(p, ProtocolParameter):
                if p.name in protocol_info:
                    if not all_elements_equal(protocol_info[p.name]):
                        const_d = {p.name: SimpleDataAdapter(protocol_info[p.name],
                                                             p.data_type, self._get_mot_float_type())}
                        return_data.update(const_d)
                else:
                    exception = 'Protocol parameter "{}" could not be resolved'.format('{}.{}'.format(m.name, p.name))
                    raise ParameterResolutionException(exception)
        return return_data

    def _get_static_data(self):
        static_data_dict = {}
        for m, p in self._model_functions_info.get_model_parameter_list():
            if isinstance(p, ModelDataParameter):
                value = self._model_functions_info.get_parameter_value('{}.{}'.format(m.name, p.name))
                if not all_elements_equal(value):
                    static_data_dict.update({p.name: SimpleDataAdapter(value, p.data_type, self._get_mot_float_type())})
        return static_data_dict

    def _get_all_kernel_source_items(self, problems_to_analyze):
        """Get the CL strings for the kernel source items for most common CL kernels in this library."""
        kernel_param_names = []
        data_struct_init = []
        data_struct_names = []

        for key, data_adapter in self._get_variable_data(problems_to_analyze).items():
            cl_data = data_adapter.get_opencl_data()

            param_name = 'var_data_' + str(key)
            data_type = data_adapter.get_data_type().raw_data_type

            if data_adapter.get_data_type().is_vector_type:
                data_type += data_adapter.get_data_type().vector_length

            kernel_param_names.append('global ' + data_type + '* ' + param_name)

            mult = cl_data.shape[1] if len(cl_data.shape) > 1 else 1
            if len(cl_data.shape) == 1 or cl_data.shape[1] == 1:
                data_struct_names.append(data_type + ' ' + param_name)
                data_struct_init.append(param_name + '[{{problem_id_name}} * {}]'.format(mult))
            else:
                data_struct_names.append('global ' + data_type + '* ' + param_name)
                data_struct_init.append(param_name + ' + {{problem_id_name}} * {}'.format(mult))

        for key, data_adapter in self._get_protocol_data().items():
            param_name = 'protocol_data_' + str(key)
            data_type = data_adapter.get_data_type().raw_data_type

            if data_adapter.get_data_type().is_vector_type:
                data_type += str(data_adapter.get_data_type().vector_length)

            kernel_param_names.append('global ' + data_type + '* ' + param_name)
            data_struct_init.append(param_name)
            data_struct_names.append('global ' + data_type + '* ' + param_name)

        for key, data_adapter in self._get_static_data().items():
            param_name = 'model_data_' + str(key)
            data_type = data_adapter.get_data_type().raw_data_type

            if data_adapter.get_data_type().is_vector_type:
                data_type += data_adapter.get_data_type().vector_length

            data_struct_init.append(param_name)

            if isinstance(data_adapter.get_opencl_data(), np.ndarray):
                kernel_param_names.append('global ' + data_type + '* ' + param_name)
                data_struct_names.append('global ' + data_type + '* ' + param_name)
            else:
                kernel_param_names.append(data_type + ' ' + param_name)
                data_struct_names.append(data_type + ' ' + param_name)

        data_struct = '''
            typedef struct{
                ''' + ('' if data_struct_names else 'constant void* place_holder;') + '''
                ''' + " ".join((name + ";\n" for name in data_struct_names)) + '''
            } ''' + self._kernel_data_struct_type + ''';
        '''

        return {'kernel_param_names': kernel_param_names,
                'data_struct_names': data_struct_names,
                'data_struct_init': data_struct_init,
                'data_struct': data_struct}

    def _get_pre_model_expression_eval_code(self):
        """The code called in the evaluation function.

        This is called after the parameters are initialized and before the model signal expression. It can call
        functions defined in _get_pre_model_expression_eval_function()

        Returns:
            str: cl code containing evaluation changes,
        """
        return ''

    def _get_pre_model_expression_eval_function(self):
        """Function used in the model evaluation generation function.

        The idea is that some implementing models may need to change some of the protocol or fixed parameters
        before they are handed over to the signal expression function. This function is called by the
        get_model_eval_function function during model evaluation function construction.

        Returns:
            str: cl function to be used in conjunction with the output of the function
                _get_pre_model_expression_eval_model()
        """

    def _set_default_dependencies(self):
        """Initialize the default dependencies.

        By default this adds dependencies for the fixed data that is used in multiple parameters.
        Additionally, if enforce weights sum to one is set, this adds the dependency on the first weight.
        """
        self._init_fixed_duplicates_dependencies()
        if self._enforce_weights_sum_to_one:
            names = ['{}.{}'.format(m.name, p.name) for (m, p) in self._model_functions_info.get_weights()]
            if len(names):
                self.fix(names[0], SimpleAssignment('max((double)1 - ({}), (double)0)'.format(' + '.join(names[1:]))))

    def _get_mot_float_type(self):
        """Get the data type for the mot_float_type"""
        if self.double_precision:
            return SimpleCLDataType.from_string('double')
        return SimpleCLDataType.from_string('float')

    def _get_weight_sum_to_one_transformation(self):
        """Returns a snippet of CL for the encode and decode functions to force the sum of the weights to 1"""
        weight_indices = []
        for (m, p) in self._model_functions_info.get_estimable_weights():
            weight_indices.append(self._model_functions_info.get_parameter_estimable_index(m, p))

        if weight_indices:
            return '''
                mot_float_type _weight_sum = ''' + ' + '.join('x[{}]'.format(index) for index in weight_indices) + ''';
                if(_weight_sum > 1.0){
                    ''' + '\n'.join('x[{}] /= _weight_sum;'.format(index) for index in weight_indices) + '''
                }
            '''
        return ''


class SampleModelBuilder(OptimizeModelBuilder):

    def __init__(self, model_name, model_tree, evaluation_model, signal_noise_model=None, problem_data=None,
                 enforce_weights_sum_to_one=True):
        """Create a new model builder for sampling purposes.

        Attributes:
            model_priors (list of mot.model_building.utils.ModelPrior): the list of model priors this class
                will also use (next to the priors defined in the parameters).
        """
        super(SampleModelBuilder, self).__init__(model_name, model_tree, evaluation_model, signal_noise_model,
                                                 problem_data, enforce_weights_sum_to_one=enforce_weights_sum_to_one)

        self._model_priors = []

        if self._enforce_weights_sum_to_one:
            weight_prior = self._get_weight_prior()
            if weight_prior:
                self._model_priors.append(weight_prior)

        for compartment in self._model_functions_info.get_model_list():
            priors = compartment.get_model_function_priors()
            if priors:
                for prior in priors:
                    self._model_priors.append(_ModelFunctionPriorToCompositeModelPrior(prior, compartment.name))

    def _init_model_information_container(self, model_tree, evaluation_model, signal_noise_model):
        """Get the model information container object.

        This is called in the __init__ to provide the new model with the correct subclass function information
        object. The rationale is that some subclasses may have additional parameters not present in optimization. For
        example, in sampling one can have priors with parameters. These parameters must be added to the model and the
        best point to do that is in the ModelFunctionsInformation object.

        Returns:
            ModelFunctionsInformation: the model function information object
        """
        return ModelFunctionsInformation(model_tree, evaluation_model, signal_noise_model, enable_prior_parameters=True)

    def build(self, problems_to_analyze=None):
        """Construct the final immutable model with the current settings.

        Returns:
            OptimizeModelInterface: an implementation an optimization model with all the current settings

        Raises:
            RuntimeError: if some of the required items are not set prior to building.
        """
        simple_optimize_model = super(SampleModelBuilder, self).build(problems_to_analyze)
        return SimpleSampleModel(simple_optimize_model,
                                 self._get_proposal_state(problems_to_analyze),
                                 self._get_log_likelihood_per_observation_function_builder(problems_to_analyze),
                                 self._is_proposal_symmetric(),
                                 self._get_log_prior_function_builder(),
                                 self._get_metropolis_hastings_state(problems_to_analyze),
                                 self._proposal_state_update_uses_variance(),
                                 self._get_proposal_logpdf_builder(),
                                 self._get_proposal_function_builder(),
                                 self._get_proposal_state_update_function_builder())

    def _get_log_prior_function_builder(self):
        def get_preliminary():
            cl_str = ''
            for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
                cl_str += p.sampling_prior.get_prior_function()

            for model_prior in self._model_priors:
                cl_str += model_prior.get_prior_function()
            return cl_str

        def get_body():
            cl_str = ''
            for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
                name = '{}.{}'.format(m.name, p.name)

                if all_elements_equal(self._lower_bounds[name]):
                    lower_bound = str(get_single_value(self._lower_bounds[name]))
                    if lower_bound == '-inf':
                        lower_bound = '-INFINITY'
                else:
                    lower_bound = 'data->var_data_lb_' + name.replace('.', '_')

                if all_elements_equal(self._upper_bounds[name]):
                    upper_bound = str(get_single_value(self._upper_bounds[name]))
                    if upper_bound == 'inf':
                        upper_bound = 'INFINITY'
                else:
                    upper_bound = 'data->var_data_ub_' + name.replace('.', '_')

                function_name = p.sampling_prior.get_prior_function_name()

                if m.get_prior_parameters(p):
                    prior_params = []
                    for prior_param in m.get_prior_parameters(p):
                        if self._model_functions_info.is_parameter_estimable(m, prior_param):
                            estimable_index = self._model_functions_info.get_parameter_estimable_index(m, prior_param)
                            prior_params.append('x[{}]'.format(estimable_index))
                        else:
                            value = self._model_functions_info.get_parameter_value(
                                '{}.{}'.format(m.name, prior_param.name))
                            if all_elements_equal(value):
                                prior_params.append(str(get_single_value(value)))
                            else:
                                prior_params.append('data->var_data_' +
                                                    '{}.{}'.format(m.name, prior_param.name).replace('.', '_'))

                    cl_str += 'prior *= {}(x[{}], {}, {}, {});\n'.format(function_name, i, lower_bound, upper_bound,
                                                                       ', '.join(prior_params))
                else:
                    cl_str += 'prior *= {}(x[{}], {}, {});\n'.format(function_name, i, lower_bound, upper_bound)

            for model_prior in self._model_priors:
                function_name = model_prior.get_prior_function_name()
                parameters = []

                for param_name in model_prior.get_function_parameters():
                    assignment_value = self._get_free_parameter_assignment_value(
                        *self._model_functions_info.get_model_parameter_by_name(param_name))
                    parameters.append(assignment_value)

                cl_str += '\tprior *= {}({});\n'.format(function_name, ', '.join(parameters))

            cl_str += '\n\treturn log(prior);'
            return cl_str

        preliminary = get_preliminary()
        body = get_body()

        def builder(address_space_parameter_vector):
            func_name = 'getLogPrior'
            prior = '''
                {preliminary}

                mot_float_type {func_name}(const void* data_void,
                                           {address_space_parameter_vector} const mot_float_type* const x){{

                    {kernel_data_struct_type}* data = ({kernel_data_struct_type}*)data_void;
                    mot_float_type prior = 1.0;

                    {body}
                }}
                '''.format(func_name=func_name, address_space_parameter_vector=address_space_parameter_vector,
                           kernel_data_struct_type=self._kernel_data_struct_type,
                           preliminary=preliminary, body=body)
            return SimpleNamedCLFunction(prior, func_name)
        return builder

    def _get_proposal_state(self, problems_to_analyze):
        np_dtype = np.float32
        if self.double_precision:
            np_dtype = np.float64

        proposal_state = []
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            for param in p.sampling_proposal.get_parameters():
                if param.adaptable:
                    value = param.default_value

                    if is_scalar(value):
                        if self._get_nmr_problems(problems_to_analyze) == 0:
                            proposal_state.append(np.full((1, 1), value, dtype=np_dtype))
                        else:
                            proposal_state.append(np.full((self._get_nmr_problems(problems_to_analyze), 1),
                                                          value, dtype=np_dtype))
                    else:
                        if len(value.shape) < 2:
                            value = np.transpose(np.asarray([value]))
                        elif value.shape[1] > value.shape[0]:
                            value = np.transpose(value)
                        else:
                            value = value

                        if problems_to_analyze is None:
                            proposal_state.append(value)
                        else:
                            proposal_state.append(value[problems_to_analyze, ...])

        proposal_state_matrix = np.concatenate([np.transpose(np.array([s]))
                                                if len(s.shape) < 2 else s for s in proposal_state], axis=1)
        return proposal_state_matrix

    def _is_proposal_symmetric(self):
        return all(p.sampling_proposal.is_symmetric() for m, p in
                   self._model_functions_info.get_estimable_parameters_list())

    def _get_proposal_logpdf_builder(self):
        def get_preliminary():
            cl_str = ''
            for _, p in self._model_functions_info.get_estimable_parameters_list():
                cl_str += p.sampling_proposal.get_proposal_logpdf_function()
            return cl_str

        def get_body():
            cl_str = 'switch(param_ind){'
            adaptable_parameter_count = 0
            for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
                cl_str += 'case ' + str(i) + ':'

                param_proposal = p.sampling_proposal
                logpdf_call = 'return ' + param_proposal.get_proposal_logpdf_function_name() + '(proposal, current'

                for param in param_proposal.get_parameters():
                    if param.adaptable:
                        logpdf_call += ', proposal_state[' + str(adaptable_parameter_count) + ']'
                        adaptable_parameter_count += 1
                    else:
                        logpdf_call += ', ' + str(param.default_value)

                logpdf_call += ');'
                cl_str += logpdf_call + "\n"
            cl_str += '} return 0;'
            return cl_str

        preliminary = get_preliminary()
        body = get_body()

        def builder(address_space_proposal_state):
            func_name = 'getProposalLogPDF'
            return_str = '''
                {preliminary}

                double {func_name}(
                    const uint param_ind,
                    const mot_float_type proposal,
                    const mot_float_type current,
                    {address_space_proposal_state} mot_float_type* const proposal_state){{

                    {body}
                }}
            '''.format(func_name=func_name, address_space_proposal_state=address_space_proposal_state,
                       body=body, preliminary=preliminary)
            return SimpleNamedCLFunction(return_str, func_name)
        return builder

    def _get_proposal_function_builder(self):
        def get_preliminary():
            cl_str = ''
            for _, p in self._model_functions_info.get_estimable_parameters_list():
                cl_str += p.sampling_proposal.get_proposal_function()
            return cl_str

        def get_body():
            cl_str = 'switch(param_ind){'
            adaptable_parameter_count = 0
            for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
                cl_str += 'case ' + str(i) + ':' + "\n\t\t\t"

                param_proposal = p.sampling_proposal
                proposal_call = 'return ' + param_proposal.get_proposal_function_name() + '(current, rng_data'

                for param in param_proposal.get_parameters():
                    if param.adaptable:
                        proposal_call += ', proposal_state[' + str(adaptable_parameter_count) + ']'
                        adaptable_parameter_count += 1
                    else:
                        proposal_call += ', ' + str(param.default_value)

                proposal_call += ');'
                cl_str += proposal_call + "\n"

            cl_str += '}\n return 0;'
            return cl_str

        preliminary = get_preliminary()
        body = get_body()

        def builder(address_space_proposal_state):
            func_name = 'getProposal'
            return_str = '''
                {preliminary}

                mot_float_type {func_name}(
                    const uint param_ind,
                    const mot_float_type current,
                    void* rng_data,
                    {address_space_proposal_state} mot_float_type* const proposal_state){{

                    {body}
                }}
            '''.format(func_name=func_name, address_space_proposal_state=address_space_proposal_state,
                       body=body, preliminary=preliminary)
            return SimpleNamedCLFunction(return_str, func_name)

        return builder

    def _get_proposal_state_update_function_builder(self):
        def get_body():
            cl_str = ''
            adaptable_parameter_count = 0
            for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
                param_proposal = p.sampling_proposal
                proposal_update_function = param_proposal.get_proposal_update_function()

                state_params = []

                for param in param_proposal.get_parameters():
                    if param.adaptable:
                        state_params.append('proposal_state + {}'.format(adaptable_parameter_count))
                        adaptable_parameter_count += 1

                if state_params:
                    if proposal_update_function.uses_jump_counters():
                        state_params.extend(['sampling_counter + {}'.format(i),
                                             'acceptance_counter + {}'.format(i)])

                    if proposal_update_function.uses_parameter_variance():
                        state_params.append('parameter_variance[{}]'.format(i))

                    cl_str += '''
                        // {param_name}
                        {update_func_name}({params});
                    '''.format(
                        update_func_name=proposal_update_function.get_function_name(param_proposal.get_parameters()),
                        params=', '.join(state_params), param_name='{}.{}'.format(m.name, p.name))
            return cl_str

        body = get_body()

        def builder(address_space):
            def get_preliminary():
                cl_str = ''
                for _, p in self._model_functions_info.get_estimable_parameters_list():
                    if p.sampling_proposal.is_adaptable():
                        cl_str += p.sampling_proposal.get_proposal_update_function().get_update_function(
                            p.sampling_proposal.get_parameters(), address_space=address_space)
                return cl_str

            preliminary = get_preliminary()

            func_name = 'updateProposalState'
            return_str = ''
            if self._proposal_state_update_uses_variance():
                return_str += '''
                    {preliminary}

                    void {func_name}({address_space} mot_float_type* const proposal_state,
                                     {address_space} ulong* const sampling_counter,
                                     {address_space} ulong* const acceptance_counter,
                                     {address_space} mot_float_type* const parameter_variance){{
                        {body}
                    }}
                '''.format(func_name=func_name, address_space=address_space, body=body, preliminary=preliminary)
            else:
                return_str += '''
                    {preliminary}

                    void {func_name}({address_space} mot_float_type* const proposal_state,
                                     {address_space} ulong* const sampling_counter,
                                     {address_space} ulong* const acceptance_counter){{
                        {body}
                    }}
                '''.format(func_name=func_name, address_space=address_space, body=body, preliminary=preliminary)

            return SimpleNamedCLFunction(return_str, func_name)

        return builder

    def _proposal_state_update_uses_variance(self):
        for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
            param_proposal = p.sampling_proposal
            proposal_update_function = param_proposal.get_proposal_update_function()

            if any(param.adaptable for param in param_proposal.get_parameters()):
                if proposal_update_function.uses_parameter_variance():
                    return True
        return False

    def _get_log_likelihood_per_observation_function_builder(self, problems_to_analyze):
        eval_function_info = self._get_model_eval_function(problems_to_analyze)
        obs_func = self._get_observation_return_function()

        param_listing = ''
        for p in self._evaluation_model.get_free_parameters():
            param_listing += self._get_param_listing_for_param(self._evaluation_model, p)

        func_name = "getLogLikelihoodPerObservation"

        preliminary = ''
        preliminary += self._evaluation_model.get_cl_dependency_code()

        preliminary += eval_function_info.get_function()
        preliminary += obs_func.get_function()

        def builder(full_likelihood):
            func = preliminary + self._evaluation_model.get_log_likelihood_per_observation_function(
                func_name, eval_function_info.get_name(),
                obs_func.get_name(), param_listing,
                full_likelihood=full_likelihood)
            return SimpleNamedCLFunction(func, func_name)

        return builder

    def _get_metropolis_hastings_state(self, problems_to_analyze):
        return DefaultMHState(self._get_nmr_problems(problems_to_analyze),
                              self.get_nmr_estimable_parameters(), self.double_precision)

    def _get_weight_prior(self):
        """Get the prior limiting the weights between 0 and 1"""
        weights = []
        for (m, p) in self._model_functions_info.get_estimable_weights():
            weights.append('{}.{}'.format(m.name, p.name))

        if len(weights) > 1:
            prior = SimpleModelPrior('''
                return (''' + ' + '.join(el.replace('.', '_') for el in weights) + ''') <= 1;
            ''', weights, 'prior_estimable_weights_sum_to_one')
            return prior
        return None


class CompositeModelFunction(ModelFunction):

    def __init__(self, model_tree, signal_noise_model=None):
        """The model function for the total constructed model.

        This combines all the functions in the model tree into one big function and exposes that function and
        its parameters.

        Args:
            model_tree (mot.model_building.trees.CompartmentModelTree): the model tree object
            signal_noise_model (mot.model_building.signal_noise_models.SignalNoiseModel): the optional signal
                noise model to use to add noise to the model prediction
        """
        self._model_tree = model_tree
        self._signal_noise_model = signal_noise_model

        self._models = list(self._model_tree.get_compartment_models())
        if self._signal_noise_model:
            self._models.append(self._signal_noise_model)
        self._parameter_model_list = list((m, p) for m in self._models for p in m.get_parameters())

    @property
    def return_type(self):
        return 'double'

    @property
    def cl_function_name(self):
        return '_composite_model_function'

    def get_parameters(self):
        return [p.get_renamed(cl_name) for m, p, cl_name in self._get_model_function_parameters()]

    def get_original_model_parameter_list(self):
        """Get the model and parameter tuples for the model out of which this composite model was constructed."""
        return [(m, p) for m, p, cl_name in self._get_model_function_parameters()]

    def get_cl_code(self):
        dependencies = []
        for model in self._models:
            dependencies.append(model.get_cl_code())

        return_str = ''
        return_str += '\n'.join(dependencies)
        return_str += self._get_model_function_cl_code()
        return return_str

    def get_free_parameters(self):
        return list([p for p in self.get_parameters() if isinstance(p, FreeParameter)])

    def _get_model_function_cl_code(self):
        """Get the CL code for the model function as build by this model.

        This returns the CL code for ONLY the model, that is it will return a function with the signature:

        .. code-block:: c

            double <func_name>(<param0>, <param1>, <param2>, ...);

        Which as output returns the model evaluated at that set of parameters.
        The model returned by this function knows nothing about the different types of parameters, it just returns
        the model equation as constructed using the model tree.

        Returns:
            str: the CL code for the model
        """
        def build_model_expression():
            tree = self._build_model_from_tree(self._model_tree, 0)

            model_expression = ''
            if self._signal_noise_model:
                noise_params = ''
                for p in self._signal_noise_model.get_free_parameters():
                    noise_params += '{}.{}'.format(self._signal_noise_model.name, p.name).replace('.', '_')
                model_expression += '{}(({}), {});'.format(self._signal_noise_model.cl_function_name,
                                                           tree, noise_params)
            else:
                model_expression += '(' + tree + ');'
            return model_expression

        def build_parameters():
            params = self._get_model_function_parameters()
            cl_parameters = []

            for m, p, name in params:
                cl_type = p.data_type.cl_type
                cl_parameters.append('{} {}'.format(cl_type, name))
            return cl_parameters

        return_str = '''
            double {func_name}(
                    {params}){{

                return {model_expression}
            }}
        '''.format(func_name=self.cl_function_name, params=indent(', \n'.join(build_parameters()), '    ' * 5)[20:],
                   model_expression=build_model_expression())
        return dedent(return_str.replace('\t', '    '))

    def _get_model_parameters_list(self):
        pass

    def _get_model_function_parameters(self):
        """Get the parameters to use in the model function.

        Returns:
            list of tuples: per parameter a tuple with (model, parameter, cl_name)
                with the cl_name the name for this parameter in this CL function and the model and parameter
                the original model and parameter.
        """
        seen_shared_params = []

        shared_params = []
        other_params = []

        for m, p in self._parameter_model_list:
            if isinstance(p, (ProtocolParameter, CurrentObservationParam)):
                if p.name not in seen_shared_params:
                    shared_params.append((m, p, p.name))
                    seen_shared_params.append(p.name)
            else:
                other_params.append((m, p, '{}_{}'.format(m.name, p.name)))
        return shared_params + other_params

    def _build_model_from_tree(self, node, depth):
        """Construct the model equation from the provided model tree.

        Args:
            node: the next to to process
            depth (int): the current tree depth

        Returns:
            str: model (sub-)equation
        """
        def model_to_string(model):
            """Convert a model to CL string."""
            param_list = []
            for param in model.get_parameters():
                if isinstance(param, (ProtocolParameter, CurrentObservationParam)):
                    param_list.append(param.name)
                else:
                    param_list.append('{}.{}'.format(model.name, param.name).replace('.', '_'))
            return model.cl_function_name + '(' + ', '.join(param_list) + ')'

        if not node.children:
            return model_to_string(node.data)
        else:
            subfuncs = []
            for child in node.children:
                if child.children:
                    subfuncs.append(self._build_model_from_tree(child, depth + 1))
                else:
                    subfuncs.append(model_to_string(child.data))

            operator = node.data
            func = (' ' + operator + ' ').join(subfuncs)

        if func[0] == '(':
            return '(' + func + ')'
        return '(' + "\n" + ("\t" * int((depth/2)+5)) + func + "\n" + ("\t" * int((depth/2)+4)) + ')'


class ModelFunctionsInformation(object):

    def __init__(self, model_tree, evaluation_model, signal_noise_model=None, enable_prior_parameters=False):
        """Contains centralized information about the model functions in the model builder parent.

        Args:
            model_tree (mot.model_building.trees.CompartmentModelTree): the model tree object
            evaluation_model (mot.model_building.evaluation_models.EvaluationModel): the evaluation model to
                use for the resulting complete model
            signal_noise_model (mot.model_building.signal_noise_models.SignalNoiseModel): the signal
                noise model to use to add noise to the model prediction
            enable_prior_parameters (boolean): adds possible prior parameters to the list of parameters in the model
        """
        self._model_tree = model_tree
        self._evaluation_model = evaluation_model
        self._signal_noise_model = signal_noise_model
        self._enable_prior_parameters = enable_prior_parameters

        self._model_list = self._get_model_list()
        self._model_parameter_list = self._get_model_parameter_list()
        self._prior_parameters_info = self._get_prior_parameters_info()

        self._check_for_double_model_names()

        self._fixed_parameters = {'{}.{}'.format(m.name, p.name): p.fixed for m, p in
                                  self.get_model_parameter_list() if isinstance(p, FreeParameter)}
        self._fixed_values = {'{}.{}'.format(m.name, p.name): p.value for m, p in self.get_free_parameters_list()}

        self._parameter_values = {'{}.{}'.format(m.name, p.name): p.value for m, p in self.get_model_parameter_list()
                                  if hasattr(p, 'value')}

    def set_parameter_value(self, parameter_name, value):
        """Set the value we will use for the given parameter.

        If the parameter is a fixed free parameter we will set the fixed value to the given value.

        Args:
            parameter_name (string): A model.param name like 'Ball.d'
            value (scalar or vector or string or AbstractParameterDependency): The value or dependency
                to fix the given parameter to. Dependency objects and strings are only value for fixed free parameters.
        """
        if parameter_name in self._fixed_parameters and self._fixed_parameters[parameter_name]:
            self._fixed_values[parameter_name] = value
        else:
            self._parameter_values[parameter_name] = value

    def get_parameter_value(self, parameter_name):
        """Get the parameter value for the given parameter. This is regardless of model fixation.

        Returns:
            float or ndarray: the value for the given parameter
        """
        if parameter_name in self._fixed_parameters and self._fixed_parameters[parameter_name]:
            return self._fixed_values[parameter_name]
        return self._parameter_values[parameter_name]

    def fix_parameter(self, parameter_name, value):
        """Fix the indicated free parameter to the given value.

        Args:
            parameter_name (string): A model.param name like 'Ball.d'
            value (scalar or vector or string or AbstractParameterDependency): The value or dependency
                to fix the given parameter to.
        """
        self._fixed_parameters[parameter_name] = True
        self._fixed_values[parameter_name] = value

    def unfix(self, parameter_name):
        """Unfix the indicated parameter

        Args:
            parameter_name (str): the name of the parameter to fix or unfix
        """
        self._fixed_parameters[parameter_name] = False

    def get_model_list(self):
        """Get the list of all the applicable model functions

        Returns:
            list of mot.model_building.model_functions.ModelFunction: the list of model functions.
        """
        return self._model_list

    def get_model_parameter_list(self):
        """Get a list of all model, parameter tuples.

        Returns:
            list of tuple: the list of tuples containing (model, parameters)
        """
        param_list = copy.copy(self._model_parameter_list)

        if self._enable_prior_parameters:
            for prior_info in self._prior_parameters_info.values():
                if prior_info:
                    param_list.extend(prior_info)

        return param_list

    def get_free_parameters_list(self, exclude_priors=False):
        """Gets the free parameters as (model, parameter) tuples from the model listing.
        This does not incorporate checking for fixed parameters.

        Args:
            exclude_priors (boolean): if we want to exclude the parameters for the priors

        Returns:
            list of tuple: the list of tuples containing (model, parameters)
        """
        free_params = list((m, p) for m, p in self._model_parameter_list if isinstance(p, FreeParameter))

        if not exclude_priors:
            if self._enable_prior_parameters:
                prior_params = []
                for m, p in free_params:
                    prior_params.extend((m, prior_p) for prior_p in m.get_prior_parameters(p)
                                        if self.is_parameter_estimable(m, p) and isinstance(prior_p, FreeParameter))
                free_params.extend(prior_params)

        return free_params

    def get_estimable_parameters_list(self, exclude_priors=False):
        """Gets a list (as model, parameter tuples) of all parameters that are estimable.

        Args:
            exclude_priors (boolean): if we want to exclude the parameters for the priors

        Returns:
            list of tuple: the list of estimable parameters
        """
        estimable_parameters = [(m, p) for m, p in self._model_parameter_list if self.is_parameter_estimable(m, p)]

        if not exclude_priors:
            if self._enable_prior_parameters:
                prior_params = []
                for m, p in estimable_parameters:
                    prior_params.extend((m, prior_p) for prior_p in m.get_prior_parameters(p) if not prior_p.fixed)
                estimable_parameters.extend(prior_params)

        return estimable_parameters

    def get_value_fixed_parameters_list(self, exclude_priors=False):
        """Gets a list (as model, parameter tuples) of all parameters that are fixed to a value.

        Args:
            exclude_priors (boolean): if we want to exclude the parameters for the priors

        Returns:
            list of tuple: the list of value fixed parameters
        """
        value_fixed_parameters = []
        for m, p in self.get_free_parameters_list(exclude_priors=exclude_priors):
            if self.is_fixed_to_value('{}.{}'.format(m.name, p.name)):
                value_fixed_parameters.append((m, p))
        return value_fixed_parameters

    def get_dependency_fixed_parameters_list(self, exclude_priors=False):
        """Gets a list (as model, parameter tuples) of all parameters that are fixed to a dependency.

        Args:
            exclude_priors (boolean): if we want to exclude the parameters for the priors

        Returns:
            list of tuple: the list of value fixed parameters
        """
        dependency_fixed_parameters = []
        for m, p in self.get_free_parameters_list(exclude_priors=exclude_priors):
            if self.is_fixed_to_dependency(m, p):
                dependency_fixed_parameters.append((m, p))
        return dependency_fixed_parameters

    def get_static_parameters_list(self):
        """Gets the static parameters (as model, parameter tuples) from the model listing."""
        static_params = list((m, p) for m, p in self.get_model_parameter_list() if isinstance(p, StaticMapParameter))

        if self._enable_prior_parameters:
            prior_params = []
            for m, p in self.get_estimable_parameters_list():
                prior_params.extend((m, prior_p) for prior_p in m.get_prior_parameters(p)
                                    if isinstance(prior_p, FreeParameter))
            static_params.extend(prior_params)

        return static_params

    def get_protocol_parameters_list(self):
        """Gets the static parameters (as model, parameter tuples) from the model listing."""
        return list((m, p) for m, p in self.get_model_parameter_list() if isinstance(p, ProtocolParameter))

    def get_model_parameter_by_name(self, parameter_name):
        """Get the parameter object of the given full parameter name in dot format.

        Args:
            parameter_name (string): the parameter name in dot format: <model>.<param>

        Returns:
            tuple: containing the (model, parameter) pair for the given parameter name
        """
        for m, p in self.get_model_parameter_list():
            if '{}.{}'.format(m.name, p.name) == parameter_name:
                return m, p
        raise ValueError('The parameter with the name "{}" could not be found in this model.'.format(parameter_name))

    def get_non_model_eval_param_listing(self):
        """Get the model, parameter tuples for all parameters that are not used in the model evaluation function.

        Basically this returns the parameters of the evaluation model.

        Returns:
            tuple: the (model, parameter) tuple for all non model evaluation parameters
        """
        listing = []
        for p in self._evaluation_model.get_parameters():
            listing.append((self._evaluation_model, p))
        return listing

    def is_fixed(self, parameter_name):
        """Check if the given (free) parameter is fixed or not

        Args:
            parameter_name (str): the name of the parameter to fix or unfix

        Returns:
            boolean: if the parameter is fixed or not (can be fixed to a value and dependency).
        """
        return parameter_name in self._fixed_parameters and self._fixed_parameters[parameter_name]

    def is_fixed_to_value(self, parameter_name):
        """Check if the given (free) parameter is fixed to a value.

        Args:
            parameter_name (str): the name of the parameter to fix or unfix

        Returns:
            boolean: if the parameter is fixed to a value or not
        """
        if self.is_fixed(parameter_name):
            return not isinstance(self._fixed_values[parameter_name], AbstractParameterDependency)
        return False

    def is_fixed_to_dependency(self, model, param):
        """Check if the given model and parameter name combo has a dependency.

        Args:
            model (mot.model_building.model_functions.ModelFunction): the model function
            param (mot.model_building.parameters.CLFunctionParameter): the parameter

        Returns:
            boolean: if the given parameter has a dependency
        """
        model_param_name = '{}.{}'.format(model.name, param.name)
        if self.is_fixed(model_param_name):
            return isinstance(self._fixed_values[model_param_name], AbstractParameterDependency)
        return False

    def is_parameter_estimable(self, model, param):
        """Check if the given model parameter is estimable.

        A parameter is estimable if it is of the Free parameter type and is not fixed.

        Args:
            model (mot.model_building.model_functions.ModelFunction): the model function
            param (mot.model_building.parameters.CLFunctionParameter): the parameter

        Returns:
            boolean: true if the parameter is estimable, false otherwise
        """
        return isinstance(param, FreeParameter) and not self.is_fixed('{}.{}'.format(model.name, param.name))

    def get_weights(self):
        """Get all the model functions/parameter tuples of the models that are a subclass of Weight

        Returns:
            list: the list of compartment models that are a subclass of Weight as (model, parameter) tuples.
        """
        weight_models = [m for m in self._model_tree.get_compartment_models() if isinstance(m, Weight)]
        weights = []
        for m in weight_models:
            for p in m.get_free_parameters():
                weights.append((m, p))
        return weights

    def get_estimable_weights(self):
        """Get all the estimable weights.

        Returns:
            list of tuples: the list of compartment models/parameter pairs for models that are a subclass of Weight
        """
        return [(m, p) for m, p in self.get_weights() if self.is_parameter_estimable(m, p)]

    def _get_model_parameter_list(self):
        """Get a list of all model, parameter tuples.

        Returns:
            list of tuple: the list of tuples containing (model, parameters)
        """
        return list((m, p) for m in self._model_list for p in m.get_parameters())

    def _get_prior_parameters_info(self):
        """Get a dictionary with the prior parameters for each of the model parameters.

        Returns:
            dict: lookup dictionary matching model names to parameter lists
        """
        prior_lookup_dict = {}
        for model in self._model_list:
            for param in model.get_free_parameters():
                prior_lookup_dict.update({
                    '{}.{}'.format(model.name, param.name): list((model, p) for p in model.get_prior_parameters(param))
                })
        return prior_lookup_dict

    def get_parameter_estimable_index(self, model, param):
        """Get the index of this parameter in the parameters list

        This returns the position of this parameter in the 'x', parameter vector in the CL kernels.

        Args:
            model (mot.model_building.model_functions.ModelFunction): the model function
            param (mot.model_building.parameters.CLFunctionParameter): the parameter

        Returns:
            int: the index of the requested parameter in the list of optimized parameters

        Raises:
            ValueError: if the given parameter could not be found as an estimable parameter.
        """
        ind = 0
        for m, p in self.get_estimable_parameters_list():
            if m.name == model.name and p.name == param.name:
                return ind
            ind += 1
        raise ValueError('The given estimable parameter "{}" could not be found in this model'.format(
            '{}.{}'.format(model.name, param.name)))

    def get_parameter_estimable_index_by_name(self, model_param_name):
        """Get the index of this parameter in the parameters list

        This returns the position of this parameter in the 'x', parameter vector in the CL kernels.

        Args:
            model_param_name (str): the model parameter name

        Returns:
            int: the index of the requested parameter in the list of optimized parameters

        Raises:
            ValueError: if the given parameter could not be found as an estimable parameter.
        """
        ind = 0
        for m, p in self.get_estimable_parameters_list():
            if '{}.{}'.format(m.name, p.name) == model_param_name:
                return ind
            ind += 1
        raise ValueError('The given estimable parameter "{}" could not be found in this model'.format(model_param_name))

    def has_parameter(self, model_param_name):
        """Check to see if the given parameter is defined in this model.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'

        Returns:
            boolean: true if the parameter is defined in this model, false otherwise.
        """
        for m, p in self.get_model_parameter_list():
            if '{}.{}'.format(m.name, p.name) == model_param_name:
                return True
        return False

    def _get_model_list(self):
        """Get the list of all the applicable model functions"""
        models = list(self._model_tree.get_compartment_models())
        models.append(self._evaluation_model)
        if self._signal_noise_model:
            models.append(self._signal_noise_model)
        return models

    def _check_for_double_model_names(self):
        models = self._model_list
        model_names = []
        for m in models:
            if m.name in model_names:
                raise DoubleModelNameException("Double model name detected in the model tree.", m.name)
            model_names.append(m.name)


class ParameterTransformedModel(OptimizeModelInterface):

    def __init__(self, model, parameter_codec):
        """Decorates the given model with parameter encoding and decoding transformations.

        This decorates a few of the given function calls with the right parameter encoding and decoding transformations
        such that both the underlying model and the calling routines are unaware that the parameters have been altered.

        Args:
            model (OptimizeModelInterface): the model to decorate
            parameter_codec (mot.model_building.utils.ParameterCodec): the parameter codec to use
        """
        self._model = model
        self._parameter_codec = parameter_codec

    def decode_parameters(self, parameters):
        """Decode the given parameters back to model space.

        Args:
            parameters (ndarray): the parameters to transform back to model space
        """
        space_transformer = CodecRunner()
        return space_transformer.decode(self._model, parameters, self._parameter_codec)

    def encode_parameters(self, parameters):
        """Decode the given parameters into optimization space

        Args:
            parameters (ndarray): the parameters to transform into optimization space
        """
        space_transformer = CodecRunner()
        return space_transformer.encode(self._model, parameters, self._parameter_codec)

    @property
    def name(self):
        return self._model.name

    @property
    def double_precision(self):
        return self._model.double_precision

    def get_free_param_names(self):
        return self._model.get_free_param_names()

    def get_kernel_data_info(self):
        return self._model.get_kernel_data_info()

    def get_nmr_problems(self):
        return self._model.get_nmr_problems()

    def get_nmr_inst_per_problem(self):
        return self._model.get_nmr_inst_per_problem()

    def get_nmr_estimable_parameters(self):
        return self._model.get_nmr_estimable_parameters()

    def get_observation_return_function(self):
        return self._model.get_observation_return_function()

    def get_pre_eval_parameter_modifier(self):
        old_modifier = self._model.get_pre_eval_parameter_modifier()
        new_fname = 'wrapped_' + old_modifier.get_name()

        code = old_modifier.get_function()
        code += self._parameter_codec.get_parameter_decode_function('_decodeParameters')
        code += '''
            void ''' + new_fname + '''(const void* const data, mot_float_type* x){
                _decodeParameters(data, x);
                ''' + old_modifier.get_name() + '''(data, x);
            }
        '''
        return SimpleNamedCLFunction(code, new_fname)

    def get_model_eval_function(self):
        return self._model.get_model_eval_function()

    def get_objective_per_observation_function(self):
        return self._model.get_objective_per_observation_function()

    def get_initial_parameters(self):
        return self.encode_parameters(self._model.get_initial_parameters())

    def get_lower_bounds(self):
        # todo add codec transform here
        return self._model.get_lower_bounds()

    def get_upper_bounds(self):
        # todo add codec transform here
        return self._model.get_upper_bounds()


class ParameterNameException(Exception):
    """Thrown when the a parameter of an given name could not be found."""
    pass


class ParameterResolutionException(Exception):
    """Thrown when a fixed parameter could not be resolved."""
    pass


class DoubleModelNameException(Exception):
    """Thrown when there are two models with the same name."""
    pass


class _ModelFunctionPriorToCompositeModelPrior(ModelFunctionPrior):

    def __init__(self, model_function_prior, compartment_name):
        """Simple prior class for easily converting the compartment priors to composite model priors."""
        self._prior_function = model_function_prior.get_prior_function()
        self._parameters = ['{}.{}'.format(compartment_name, p)
                            for p in model_function_prior.get_function_parameters()]
        self._function_name = model_function_prior.get_prior_function_name()

    def get_prior_function(self):
        return self._prior_function

    def get_function_parameters(self):
        return self._parameters

    def get_prior_function_name(self):
        return self._function_name


class SimpleKernelDataInfo(KernelDataInfo):

    def __init__(self, data, kernel_parameters, kernel_struct, struct_type, init_format_str):
        """Simple kernel data information container.

        Args:
            data (list): list with ndarrays
            kernel_parameters (list of str): the kernel parameters for each of the data elements
            kernel_struct (str): the kernel structure containing all the data in the kernel
            struct_type (str): the type of the kernel structure
            init_format_str (str): the kernel data structure initialization string. This is used to
                format the init string using the python string format function.
        """
        self._data = data
        self._kernel_parameters = kernel_parameters
        self._kernel_struct = kernel_struct
        self._init_format_str = init_format_str
        self._struct_type = struct_type

    def get_data(self):
        return self._data

    def get_kernel_data_struct(self):
        return self._kernel_struct

    def get_kernel_parameters(self):
        return self._kernel_parameters

    def get_kernel_data_struct_initialization(self, variable_name, problem_id_name='gid'):
        return self._init_format_str.format(variable_name=variable_name, problem_id_name=problem_id_name)

    def get_kernel_data_struct_type(self):
        return self._struct_type


class SimpleOptimizeModel(OptimizeModelInterface):

    def __init__(self, used_problem_indices,
                 name, double_precision, free_param_names, kernel_data_info, nmr_problems, nmr_inst_per_problem,
                 nmr_estimable_parameters, initial_parameters, pre_eval_parameter_modifier, eval_function,
                 observation_return_function, objective_per_observation_function,
                 lower_bounds, upper_bounds):
        self.used_problem_indices = used_problem_indices
        self._name = name
        self._double_precision = double_precision
        self._free_param_names = free_param_names
        self._kernel_data_info = kernel_data_info
        self._nmr_problems = nmr_problems
        self._nmr_inst_per_problem = nmr_inst_per_problem
        self._nmr_estimable_parameters = nmr_estimable_parameters
        self._initial_parameters = initial_parameters
        self._pre_eval_parameter_modifier = pre_eval_parameter_modifier
        self._eval_function = eval_function
        self._observation_return_function = observation_return_function
        self._objective_per_observation_function = objective_per_observation_function
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    @property
    def name(self):
        return self._name

    @property
    def double_precision(self):
        return self._double_precision

    def get_free_param_names(self):
        return self._free_param_names

    def get_kernel_data_info(self):
        return self._kernel_data_info

    def get_nmr_problems(self):
        return self._nmr_problems

    def get_nmr_inst_per_problem(self):
        return self._nmr_inst_per_problem

    def get_nmr_estimable_parameters(self):
        return self._nmr_estimable_parameters

    def get_pre_eval_parameter_modifier(self):
        return self._pre_eval_parameter_modifier

    def get_model_eval_function(self):
        return self._eval_function

    def get_observation_return_function(self):
        return self._observation_return_function

    def get_objective_per_observation_function(self):
        return self._objective_per_observation_function

    def get_initial_parameters(self):
        return self._initial_parameters

    def get_lower_bounds(self):
        return self._lower_bounds

    def get_upper_bounds(self):
        return self._upper_bounds


class SimpleSampleModel(SampleModelInterface):

    def __init__(self, wrapped_optimize_model, proposal_state, ll_per_obs_func_builder,
                 is_proposal_symmetric, log_prior_function_builder, metropolis_hastings_state,
                 proposal_state_update_uses_variance, proposal_logpdf_builder, proposal_function_builder,
                 proposal_state_update_function_builder):
        self._wrapped_optimize_model = wrapped_optimize_model
        self._proposal_state = proposal_state
        self._ll_per_obs_func_builder = ll_per_obs_func_builder
        self._is_proposal_symmetric = is_proposal_symmetric
        self._log_prior_function_builder = log_prior_function_builder
        self._metropolis_hastings_state = metropolis_hastings_state
        self._proposal_state_update_uses_variance = proposal_state_update_uses_variance
        self._proposal_logpdf_builder = proposal_logpdf_builder
        self._proposal_function_builder = proposal_function_builder
        self._proposal_state_update_function_builder = proposal_state_update_function_builder

    @property
    def name(self):
        return self._wrapped_optimize_model.name

    @property
    def double_precision(self):
        return self._wrapped_optimize_model.double_precision

    def get_free_param_names(self):
        return self._wrapped_optimize_model.get_free_param_names()

    def get_kernel_data_info(self):
        return self._wrapped_optimize_model.get_kernel_data_info()

    def get_nmr_problems(self):
        return self._wrapped_optimize_model.get_nmr_problems()

    def get_nmr_inst_per_problem(self):
        return self._wrapped_optimize_model.get_nmr_inst_per_problem()

    def get_nmr_estimable_parameters(self):
        return self._wrapped_optimize_model.get_nmr_estimable_parameters()

    def get_pre_eval_parameter_modifier(self):
        return self._wrapped_optimize_model.get_pre_eval_parameter_modifier()

    def get_model_eval_function(self):
        return self._wrapped_optimize_model.get_model_eval_function()

    def get_observation_return_function(self):
        return self._wrapped_optimize_model.get_observation_return_function()

    def get_objective_per_observation_function(self):
        return self._wrapped_optimize_model.get_objective_per_observation_function()

    def get_initial_parameters(self):
        return self._wrapped_optimize_model.get_initial_parameters()

    def get_lower_bounds(self):
        return self._wrapped_optimize_model.get_lower_bounds()

    def get_upper_bounds(self):
        return self._wrapped_optimize_model.get_upper_bounds()

    def get_proposal_state(self):
        return self._proposal_state

    def get_log_likelihood_per_observation_function(self, full_likelihood=True):
        return self._ll_per_obs_func_builder(full_likelihood)

    def is_proposal_symmetric(self):
        return self._is_proposal_symmetric

    def get_proposal_logpdf(self, address_space_proposal_state='private'):
        return self._proposal_logpdf_builder(address_space_proposal_state)

    def get_proposal_function(self, address_space_proposal_state='private'):
        return self._proposal_function_builder(address_space_proposal_state)

    def get_proposal_state_update_function(self, address_space='private'):
        return self._proposal_state_update_function_builder(address_space)

    def proposal_state_update_uses_variance(self):
        return self._proposal_state_update_uses_variance

    def get_log_prior_function(self, address_space_parameter_vector='private'):
        return self._log_prior_function_builder(address_space_parameter_vector)

    def get_metropolis_hastings_state(self):
        return self._metropolis_hastings_state
