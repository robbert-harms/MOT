import numpy as np
import copy
from mot.cl_data_type import CLDataType
from mot.cl_routines.mapping.calc_dependent_params import CalculateDependentParameters
from mot.model_building.cl_functions.model_functions import Weight
from mot.model_building.cl_functions.parameters import CurrentObservationParam, StaticMapParameter, ProtocolParameter, \
    ModelDataParameter, FreeParameter
from mot.model_building.data_adapter import SimpleDataAdapter
from mot.model_building.parameter_functions.dependencies import SimpleAssignment
from mot.model_interfaces import OptimizeModelInterface, SampleModelInterface
from mot.utils import TopologicalSort, is_scalar, all_elements_equal, get_single_value

__author__ = 'Robbert Harms'
__date__ = "2014-03-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class OptimizeModelBuilder(OptimizeModelInterface):

    def __init__(self, name, model_tree, evaluation_model, signal_noise_model=None, problem_data=None,
                 enforce_weights_sum_to_one=True):
        """Create a new model builder that can construct an optimization model using parts.

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

        Attributes:
            problems_to_analyze (list): the list with problems we want to analyze. Suppose we have a few thousands
                problems defined in this model, but we want to run the optimization only on a few problems. By setting
                this attribute to a list of problem indices, only those problems will be analyzed.
            use_parameter_transformations (boolean): set to False to disable the parameter transformations.
                This basically sets the encode and decode functions to the identity function.
        """
        super(OptimizeModelBuilder, self).__init__()
        self._name = name
        self._model_tree = model_tree
        self._evaluation_model = evaluation_model
        self._signal_noise_model = signal_noise_model

        self._enforce_weights_sum_to_one = enforce_weights_sum_to_one

        self.use_parameter_transformations = True
        self._double_precision = False

        self._dependency_store = _DependencyStore()
        self._model_functions_info = self._init_model_information_container(
            self._dependency_store, model_tree, evaluation_model, signal_noise_model)

        self._parameters_dot_to_bar = {}
        self._post_optimization_modifiers = []
        self.problems_to_analyze = None

        # The values to use for the parameters, this is the place subclasses should use for their information.
        # At init this is filled with the values as defined in the parameters.
        self._parameter_values = {'{}.{}'.format(m.name, p.name): p.value for m, p in
                                  self._model_functions_info.get_free_parameters_list() +
                                  self._model_functions_info.get_static_parameters_list()}

        self._lower_bounds = {'{}.{}'.format(m.name, p.name): p.lower_bound for m, p in
                              self._model_functions_info.get_free_parameters_list()}

        self._upper_bounds = {'{}.{}'.format(m.name, p.name): p.upper_bound for m, p in
                              self._model_functions_info.get_free_parameters_list()}

        self._problem_data = None
        if problem_data:
            self.set_problem_data(problem_data)

        for m, p in self._model_functions_info.get_model_parameter_list():
            self._parameters_dot_to_bar.update({m.name + '.' + p.name: m.name + '_' + p.name})

        self._set_default_dependencies()

    def _init_model_information_container(self, dependency_store, model_tree, evaluation_model, signal_noise_model):
        """Get the model information container object.

        This is called in the __init__ to provide the new model with the correct subclass function information
        object. The rationale is that some subclasses may have additional parameters not present in optimization. For
        example, in sampling one can have priors with parameters. These parameters must be added to the model and the
        best point to do that is in the ModelFunctionsInformation object.

        Returns:
            ModelFunctionsInformation: the model function information object
        """
        return ModelFunctionsInformation(dependency_store, model_tree, evaluation_model, signal_noise_model)

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
            value (scalar or vector): The value to fix the given parameter to

        Returns:
            Returns self for chainability
        """
        self._parameter_values[model_param_name] = value
        self._model_functions_info.set_fixed_to_value(model_param_name, True)
        return self

    def init(self, model_param_name, value):
        """Init the given model.param to the given value.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'
            value (scalar or vector): The value to initialize the given parameter to

        Returns:
            Returns self for chainability
        """
        self._parameter_values[model_param_name] = value
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

    def unfix(self, model_param_name):
        """Unfix the given model.param

        Args:
            model_param_name (string): A model.param name like 'Ball.d'

        Returns:
            Returns self for chainability
        """
        self._model_functions_info.set_fixed_to_value(model_param_name, False)
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
            problem_data (ProblemData):
                The container for the problem data we will use for this model.

        Returns:
            Returns self for chainability
        """
        self._problem_data = problem_data
        if self._problem_data.noise_std is not None:
            self._parameter_values['{}.{}'.format(
                self._evaluation_model.name,
                self._evaluation_model.get_noise_std_param_name())] = self._problem_data.noise_std
        return self

    def add_parameter_dependency(self, parameter_name, dependency):
        """Adds a dependency rule to this model. The dependency is supposed to be a ParameterDependency object.

        The dependencies are executed in the same order as they were added to this model.

        Args:
            parameter_name (String): The parameter on which the dependency is applied
            dependency (ParameterDependency): The dependency rule, an ParameterDependency object
        """
        if parameter_name not in self._parameters_dot_to_bar:
            raise ParameterNameException("The parameter name \"{}\" can not be "
                                         "found in the model listing.".format(parameter_name))
        self._dependency_store.set_dependency(parameter_name, dependency)
        return self

    def add_parameter_dependencies(self, dependencies):
        """Adds a list of dependency objects. The order of the dependencies matter (order of parameter instantiation).

        Args:
            dependencies: the dependency rules, an tuple list like ((name, ParameterDependency object),)
        """
        for name, d in dependencies:
            self.add_parameter_dependency(name, d)
        return self

    def add_post_optimization_modifier(self, model_param_name, mod_routine):
        """Add a modification function that can update the results of model optimization.

        The mod routine should be a function accepting a dictionary as input and should return a single map of
        the same dimension as the maps in the dictionary. The idea is that the mod_routine function gets the
        result dictionary from the optimization routine and calculates a new map.

        This map is returned and the dictionary is updated with the returned map as value and the here given
        model_param_name as key.

        It is possible to add more than one modifier function. In that case, they are called in the order they
        were appended to this model.

        Args:
            model_param_name (str): the parameter to which to add the modification routine
            mod_routine (python function): the callback function to apply on the results of the referenced parameter.
        """
        self._post_optimization_modifiers.append((model_param_name, mod_routine))
        return self

    def add_post_optimization_modifiers(self, modifiers):
        """Add a list of modifier functions.

        The same as add_post_optimization_modifier() except that it accepts a list of lists. Every element in the list
        should be a tuple like (model_param_name, mod_routine)

        Args:
            modifiers (tuple or list): The list of modifiers to add (in order).

        """
        self._post_optimization_modifiers.extend(modifiers)

    def get_required_protocol_names(self):
        """Get a list with the constant data names that are needed for this model to work.

        For example, an implementing diffusion MRI model might require the presence of the protocol parameter
        'g' and 'b'. This function should then return ('g', 'b').

        Returns:
            list: A list of columns names that need to be present in the protocol
        """
        return list(set([p.name for m, p in self._model_functions_info.get_model_parameter_list() if
                         isinstance(p, ProtocolParameter)]))

    def get_optimization_output_param_names(self):
        """See super class for details"""
        items = ['{}.{}'.format(m.name, p.name) for m, p in self._model_functions_info.get_free_parameters_list()]
        items.extend(name for name, _ in self._post_optimization_modifiers)
        return items

    def get_optimized_param_names(self):
        """See super class for details"""
        return ['{}.{}'.format(m.name, p.name) for m, p in
                self._model_functions_info.get_estimable_parameters_list()]

    def get_nmr_problems(self):
        """See super class for details"""
        if self.problems_to_analyze is None:
            return self._problem_data.get_nmr_problems()
        return len(self.problems_to_analyze)

    def get_nmr_inst_per_problem(self):
        """See super class for details"""
        return self._problem_data.get_nmr_inst_per_problem()

    def get_nmr_estimable_parameters(self):
        """See super class for details"""
        return len(self.get_optimized_param_names())

    def get_data(self):
        """See super class for details"""
        data = []
        for data_dict in [self._get_variable_data(), self._get_protocol_data(), self._get_static_data()]:
            for el in data_dict.values():
                data.append(el.get_opencl_data())
        return data

    def get_kernel_data_struct(self, device):
        """See super class for details"""
        return self._get_all_kernel_source_items(device)['data_struct']

    def get_kernel_param_names(self, device):
        """See super class for details"""
        return self._get_all_kernel_source_items(device)['kernel_param_names']

    def get_kernel_data_struct_initialization(self, device, variable_name, problem_id_name='gid'):
        """See super class for details"""
        data_struct_init = self._get_all_kernel_source_items(device, problem_id_name)['data_struct_init']
        struct_code = '0'
        if data_struct_init:
            struct_code = ', '.join(data_struct_init)
        return self.get_kernel_data_struct_type() + ' ' + variable_name + ' = {' + struct_code + '};'

    def get_kernel_data_struct_type(self):
        """Get the CL type of the kernel datastruct.

        Returns:
            str: the CL type of the data struct
        """
        return '_model_data'

    def get_parameter_decode_function(self, fname='decodeParameters'):
        func = '''
            void ''' + fname + '''(const void* data_void, mot_float_type* x){
        '''
        if self.use_parameter_transformations:
            func += self.get_kernel_data_struct_type() + \
                    '* data = (' + self.get_kernel_data_struct_type() + '*)data_void;'
            for d in self._get_parameter_transformations()[1]:
                func += "\n" + "\t" * 4 + d.format('x')

        if self._enforce_weights_sum_to_one:
            func += self._get_weight_sum_to_one_transformation()

        return func + '''
            }
        '''

    def get_parameter_encode_function(self, fname='encodeParameters'):
        func = '''
            void ''' + fname + '''(const void* data_void, mot_float_type* x){
        '''

        if self._enforce_weights_sum_to_one:
            func += self._get_weight_sum_to_one_transformation()

        if self.use_parameter_transformations:
            func += self.get_kernel_data_struct_type() + \
                    '* data = (' + self.get_kernel_data_struct_type() + '*)data_void;'
            for d in self._get_parameter_transformations()[0]:
                func += "\n" + "\t" * 4 + d.format('x')

        return func + '''
            }
        '''

    def get_initial_parameters(self, results_dict=None):
        """When overriding this function, please note that it should adhere to the attribute problems_to_analyze.

        Args:
            results_dict (dict): the initialization settings for the specific parameters.
                The number of items per dictionary item should match the number of problems to analyze.
        """
        np_dtype = np.float32
        if self.double_precision:
            np_dtype = np.float64

        starting_points = []
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            param_name = '{}.{}'.format(m.name, p.name)
            value = self._parameter_values[param_name]

            if results_dict and param_name in results_dict:
                starting_points.append(results_dict[m.name + '.' + p.name])
            elif is_scalar(value):
                if self.get_nmr_problems() == 0:
                    starting_points.append(np.full((1, 1), value, dtype=np_dtype))
                else:
                    starting_points.append(np.full((self.get_nmr_problems(), 1), value, dtype=np_dtype))
            else:
                if len(value.shape) < 2:
                    value = np.transpose(np.asarray([value]))
                elif value.shape[1] > value.shape[0]:
                    value = np.transpose(value)
                else:
                    value = value

                if self.problems_to_analyze is None:
                    starting_points.append(value)
                else:
                    starting_points.append(value[self.problems_to_analyze, ...])

        starting_points = np.concatenate([np.transpose(np.array([s]))
                                          if len(s.shape) < 2 else s for s in starting_points], axis=1)

        data_adapter = SimpleDataAdapter(starting_points, CLDataType.from_string('mot_float_type'),
                                         self._get_mot_float_type())
        return data_adapter.get_opencl_data()

    def get_lower_bounds(self):
        """See super class for details"""
        return list(self._lower_bounds['{}.{}'.format(m.name, p.name)] for m, p in
                    self._model_functions_info.get_estimable_parameters_list())

    def get_upper_bounds(self):
        """See super class for details"""
        return list(self._upper_bounds['{}.{}'.format(m.name, p.name)] for m, p in
                    self._model_functions_info.get_estimable_parameters_list())

    def set_initial_parameters(self, initial_params):
        """Update the initial parameters for this model by the given values.

        This only affects free non-fixed parameters.

        Args:
            initial_params (dict): a dictionary containing as keys full parameter names (<model>.<param>) and as values
                numbers or arrays to be used as starting point
        """
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            param_name = '{}.{}'.format(m.name, p.name)

            if param_name in initial_params:
                self._parameter_values[param_name] = initial_params[param_name]

        return self

    def get_observation_return_function(self, func_name='getObservation'):
        if self._problem_data.observations.shape[1] < 2:
            return '''
                mot_float_type ''' + func_name + '''(const void* const data, const int observation_index){
                    return ((''' + self.get_kernel_data_struct_type() + '''*)data)->var_data_observations;
                }
            '''

        return '''
            mot_float_type ''' + func_name + '''(const void* const data, const int observation_index){
                return ((''' + \
                    self.get_kernel_data_struct_type() + '''*)data)->var_data_observations[observation_index];
            }
        '''

    def get_model_eval_function(self, func_name='evaluateModel'):
        noise_func_name = func_name + '_signalNoiseModel'
        func = self._get_model_functions_cl_code(noise_func_name)

        pre_model_function = self._get_pre_model_expression_eval_function()
        if pre_model_function:
            func += pre_model_function

        func += '''
            mot_float_type ''' + func_name + \
                '(const void* const void_data, const mot_float_type* const x, const int observation_index){' + "\n"
        func += self.get_kernel_data_struct_type() + '* data = (' + self.get_kernel_data_struct_type() + '*)void_data;'

        func += self._get_parameters_listing(
            exclude_list=[m.name + '_' + p.name for (m, p) in
                          self._model_functions_info.get_non_model_tree_param_listing()])

        if self._signal_noise_model:
            noise_params_listing = ''
            for p in self._signal_noise_model.get_free_parameters():
                noise_params_listing += "\t" * 4 + self._get_param_listing_for_param(self._signal_noise_model, p)
            func += "\n"
            func += noise_params_listing

        pre_model_code = self._get_pre_model_expression_eval_code()
        if pre_model_code:
            func += self._get_pre_model_expression_eval_code()

        func += "\n" + "\t"*4 + 'return ' + str(self._construct_model_expression(noise_func_name))
        func += "\n\t\t\t}"
        return func

    def get_objective_function(self, func_name="calculateObjective"):
        inst_per_problem = self.get_nmr_inst_per_problem()
        eval_func_name = func_name + '_evaluateModel'
        obs_func_name = func_name + '_getObservation'

        param_listing = ''
        for p in self._evaluation_model.get_free_parameters():
            param_listing += self._get_param_listing_for_param(self._evaluation_model, p)

        func = ''
        func += self._evaluation_model.get_cl_dependency_headers()
        func += self._evaluation_model.get_cl_dependency_code()

        func += self.get_model_eval_function(eval_func_name)
        func += self.get_observation_return_function(obs_func_name)
        func += str(self._evaluation_model.get_objective_function(func_name, inst_per_problem, eval_func_name,
                                                                  obs_func_name, param_listing))
        return str(func)

    def get_objective_per_observation_function(self, func_name="getObjectiveInstanceValue"):
        inst_per_problem = self.get_nmr_inst_per_problem()
        eval_func_name = func_name + '_evaluateModel'
        obs_func_name = func_name + '_getObservation'

        param_listing = ''
        for p in self._evaluation_model.get_free_parameters():
            param_listing += self._get_param_listing_for_param(self._evaluation_model, p)

        func = ''
        func += self._evaluation_model.get_cl_dependency_headers()
        func += self._evaluation_model.get_cl_dependency_code()

        func += self.get_model_eval_function(eval_func_name)
        func += self.get_observation_return_function(obs_func_name)
        func += str(self._evaluation_model.get_objective_per_observation_function(
            func_name, inst_per_problem, eval_func_name, obs_func_name, param_listing))
        return str(func)

    def finalize_optimization_results(self, results_dict):
        """This adds the final optimization maps to the results dictionary.

        Steps in finalizing the results dict:
            1) It first adds the maps for the dependent and fixed parameters
            2) Second it adds the extra maps defined in the models itself.
            3) Third it loops through the post_optimization_modifiers callback functions for the final updates.
            4) Adds additional maps defined in this model subclass

        For more documentation see the base method.

        Args:
            results_dict (dict): the dictionary with the results (the keys are the parameter names, the values are the
                1d parameter lists)

        """
        self._add_dependent_parameter_maps(results_dict)
        self._add_fixed_parameter_maps(results_dict)

        for model in self._model_functions_info.get_model_list():
            results_dict.update(model.get_extra_results_maps(results_dict))

        for name, routine in self._post_optimization_modifiers:
            results_dict[name] = routine(results_dict)

        self._add_finalizing_result_maps(results_dict)

        return results_dict

    def _add_fixed_parameter_maps(self, results_dict):
        """In place add complete maps for the fixed parameters."""
        param_lists = self._get_parameter_type_lists()
        fixed_params = param_lists['fixed']
        for (m, p) in fixed_params:
            if not self._model_functions_info.is_parameter_fixed_to_dependency(m, p):
                name = m.name + '.' + p.name
                value = self._parameter_values['{}.{}'.format(m.name, p.name)]

                if is_scalar(value):
                    results_dict.update({name: np.tile(np.array([value]), (self.get_nmr_problems(),))})
                else:
                    if self.problems_to_analyze is not None:
                        value = value[self.problems_to_analyze, ...]
                    results_dict.update({name: value})

    def _add_dependent_parameter_maps(self, results_dict):
        """In place add complete maps for the dependent parameters."""
        param_lists = self._get_parameter_type_lists()
        if len(param_lists['dependent']):
            func = ''
            func += self._get_fixed_parameters_listing(param_lists['fixed'])
            func += self._get_estimable_parameters_listing(param_lists['estimable'])
            func += self._get_dependent_parameters_listing(param_lists['dependent'])

            estimable_params = [m.name + '.' + p.name for m, p in param_lists['estimable']]
            estimated_parameters = [results_dict[k] for k in estimable_params]

            dependent_parameter_names = [(m.name + '_' + p.name, m.name + '.' + p.name)
                                         for m, p in param_lists['dependent']]

            cpd = CalculateDependentParameters(double_precision=self.double_precision)
            dependent_parameters = cpd.calculate(self, estimated_parameters, func, dependent_parameter_names)

            results_dict.update(dependent_parameters)

    def _get_parameter_transformations(self):
        dep_list = {}
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            dep_list.update({(m, p): (tuple(dep) for dep in p.parameter_transform.dependencies)})

        dep_list = TopologicalSort(dep_list).get_flattened()

        dec_func_list = []
        enc_func_list = []
        for m, p in dep_list:
            name = m.name + '.' + p.name
            parameter = p
            ind = self._model_functions_info.get_parameter_estimable_index(m, p)
            transform = parameter.parameter_transform

            dependency_names = []
            for dep in transform.dependencies:
                dep_ind = self._model_functions_info.get_parameter_estimable_index(dep[0], dep[1])
                dependency_names.append('{0}[' + str(dep_ind) + ']')

            if all_elements_equal(self._lower_bounds[name]):
                lower_bound = str(get_single_value(self._lower_bounds[name]))
            else:
                lower_bound = 'data->var_data_lb_' + name.replace('.', '_')

            if all_elements_equal(self._upper_bounds[name]):
                upper_bound = str(get_single_value(self._upper_bounds[name]))
            else:
                upper_bound = 'data->var_data_ub_' + name.replace('.', '_')

            s = '{0}[' + str(ind) + '] = ' + transform.get_cl_decode().create_assignment(
                '{0}[' + str(ind) + ']', lower_bound, upper_bound, dependency_names) + ';'

            dec_func_list.append(s)

            s = '{0}[' + str(ind) + '] = ' + transform.get_cl_encode().create_assignment(
                '{0}[' + str(ind) + ']', lower_bound, upper_bound, dependency_names) + ';'

            enc_func_list.append(s)

        return tuple(reversed(enc_func_list)), dec_func_list

    def _add_finalizing_result_maps(self, results_dict):
        """Add some final results maps to the results dictionary.

        This called by the function finalize_optimization_results() as last call to add more maps.

        Args:
            results_dict (args): the results from model optmization. We are to modify this in-place.
        """

    def _transform_observations(self, observations):
        """Apply a transformation on the observations before fitting.

        This function is called by get_problems_var_data() just before the observations are handed over to the
        CL routine, and just after the the list has been (optionally) limited with self.problems_to_analyze.

        To implement any behaviour here, you can override this function and add behaviour that changes the observations.

        Args:
            observations (ndarray): the 2d matrix with the observations. This is the list of
                observations *after* the list has been (optionally) limited with self.problems_to_analyze.

        Returns:
            observations (ndarray): a 2d matrix of the same shape as the input. This should hold the transformed data.
        """
        return observations

    def _construct_model_expression(self, noise_func_name):
        """Construct the model signel expression. This is supposed to be used in get_model_eval_function.

        Args:
            noise_func_name (str): the name of the noise function.
        """
        func = ''
        if self._signal_noise_model:
            noise_params_func = ''
            for p in self._signal_noise_model.get_free_parameters():
                noise_params_func += ', ' + self._signal_noise_model.name + '_' + p.name

            func += noise_func_name + '((' + self._build_model_from_tree(self._model_tree, 0) + ')' + \
                    noise_params_func + ');'
        else:
            func += '(' + self._build_model_from_tree(self._model_tree, 0) + ');'
        return func

    def _build_model_from_tree(self, node, depth):
        if not node.children:
            return self._model_to_string(node.data)
        else:
            subfuncs = []
            for child in node.children:
                if child.children:
                    subfuncs.append(self._build_model_from_tree(child, depth+1))
                else:
                    subfuncs.append(self._model_to_string(child.data))

            operator = node.data
            func = (' ' + operator + ' ').join(subfuncs)

        if func[0] == '(':
            return '(' + func + ')'
        return '(' + "\n" + ("\t" * int((depth/2)+5)) + func + "\n" + ("\t" * int((depth/2)+4)) + ')'

    def _model_to_string(self, model):
        """Convert a model to CL string."""
        param_list = []
        for param in model.parameter_list:
            if isinstance(param, ProtocolParameter):
                param_list.append(param.name)
            elif isinstance(param, ModelDataParameter):
                value = self._parameter_values['{}.{}'.format(model.name, param.name)]
                if all_elements_equal(value):
                    param_list.append(str(get_single_value(value)))
                else:
                    param_list.append('data->model_data_' + param.name)
            elif isinstance(param, StaticMapParameter):
                static_map_value = self._get_static_map_value(model, param)
                if all_elements_equal(static_map_value):
                    param_list.append(str(get_single_value(static_map_value)))
                else:
                    if len(static_map_value.shape) > 1 \
                            and static_map_value.shape[1] == self._problem_data.observations.shape[1]:
                        param_list.append('data->var_data_' + model.name + '_' + param.name + '[observation_index]')
                    else:
                        param_list.append('data->var_data_' + model.name + '_' + param.name)
            elif isinstance(param, CurrentObservationParam):
                param_list.append('data->var_data_observations[observation_index]')
            else:
                param_list.append(model.name + '_' + param.name)

        return model.cl_function_name + '(' + ', '.join(param_list) + ')'

    def _get_model_functions_cl_code(self, noise_func_name):
        """Get the model functions CL. This is used in get_model_eval_function()."""
        cl_code = ''
        for compartment_model in self._model_tree.get_compartment_models():
            cl_code += compartment_model.get_cl_header() + "\n"
            cl_code += compartment_model.get_cl_code() + "\n"

        if self._signal_noise_model:
            cl_code += self._signal_noise_model.get_signal_function(noise_func_name)
        return cl_code

    def _get_parameters_listing(self, exclude_list=()):
        """Get the CL code for the parameter listing, this goes on top of the evaluate function.

        Args:
            exclude_list: an optional list containing parameters to exclude from the listing.
             This should contain full parameter names like: <model_name>_<param_name>

        Returns:
            An CL string that contains all the parameters as primitive data types.
        """
        func = ''
        param_lists = self._get_parameter_type_lists()
        func += self._get_protocol_parameters_listing(param_lists['protocol'], exclude_list=exclude_list)
        func += self._get_fixed_parameters_listing(param_lists['fixed'], exclude_list=exclude_list)
        func += self._get_estimable_parameters_listing(param_lists['estimable'], exclude_list=exclude_list)
        func += self._get_dependent_parameters_listing(param_lists['dependent'], exclude_list=exclude_list)
        return str(func)

    def _get_estimable_parameters_listing(self, param_list=None, exclude_list=()):
        """Get the parameter listing for the free parameters.

        For performance reasons, the parameter list should already be given.
            If not given it is calculated using:
                self._get_parameter_type_lists()['estimable']

        Args:
            param_list: the list with the estimable parameters
            exclude_list: a list of parameters to exclude from this listing
        """
        if param_list is None:
            param_list = self._get_parameter_type_lists()['estimable']

        func = ''
        estimable_param_counter = 0
        for m, p in param_list:
            name = m.name + '_' + p.name
            if name not in exclude_list:
                data_type = p.data_type.cl_type
                assignment = 'x[' + str(estimable_param_counter) + ']'
                func += "\t"*4 + data_type + ' ' + name + ' = ' + assignment + ';' + "\n"
                estimable_param_counter += 1
        return func

    def _get_protocol_parameters_listing(self, param_list=None, exclude_list=()):
        """Get the parameter listing for the protocol parameters.

        For performance reasons, the parameter list should already be given.
            If not given it is calculated using:
                self._get_parameter_type_lists()['protocol']

        Args:
            param_list: the list with the protocol parameters
            exclude_list: a list of parameters to exclude from this listing
        """
        protocol_info = self._problem_data.protocol
        if param_list is None:
            param_list = self._get_parameter_type_lists()['protocol']

        const_params_seen = []
        func = ''
        for m, p in param_list:
            if (m.name + '_' + p.name) not in exclude_list:
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

    def _get_fixed_parameters_listing(self, param_list=None, exclude_list=()):
        """Get the parameter listing for the fixed parameters.

        For performance reasons, the fixed parameter list should already be given.
            If not given it is calculated using:
                self._get_parameter_type_lists()['fixed']

        Args:
            dependent_param_list: the list list of fixed params
            exclude_list: a list of parameters to exclude from this listing
        """
        if param_list is None:
            param_list = self._get_parameter_type_lists()['fixed']

        func = ''
        for m, p in param_list:
            name = m.name + '_' + p.name
            if name not in exclude_list:
                data_type = p.data_type.raw_data_type
                value = self._parameter_values['{}.{}'.format(m.name, p.name)]

                if all_elements_equal(value):
                    assignment = '(' + data_type + ')' + str(float(get_single_value(value)))
                else:
                    assignment = '(' + data_type + ') data->var_data_' + m.name + '_' + p.name
                func += "\t"*4 + data_type + ' ' + name + ' = ' + assignment + ';' + "\n"
        return func

    def _get_dependent_parameters_listing(self, dependent_param_list=None, exclude_list=()):
        """Get the parameter listing for the dependent parameters.

        For performance reasons, the dependent parameter list should already be given.
            If not given it is calculated using:
                self._get_parameter_type_lists()['dependent']

        Args:
            dependent_param_list: the list list of dependent params
            exclude_list: a list of parameters to exclude from this listing, note that this will only exclude the
                definition of the parameter, not the dependency code.
        """
        if dependent_param_list is None:
            dependent_param_list = self._get_parameter_type_lists()['dependent']
        func = ''
        for m, p in dependent_param_list:
            pd = self._dependency_store.get_dependency(m.name + '.' + p.name)
            if pd.pre_transform_code:
                func += "\t"*4 + self._convert_parameters_dot_to_bar(pd.pre_transform_code)

            assignment = self._convert_parameters_dot_to_bar(pd.assignment_code)
            name = m.name + '_' + p.name
            data_type = p.data_type.raw_data_type

            if self._model_functions_info.is_parameter_fixed_to_dependency(m, p):
                if (m.name + '_' + p.name) not in exclude_list:
                    func += "\t"*4 + data_type + ' ' + name + ' = ' + assignment + ';' + "\n"
            else:
                func += "\t"*4 + name + ' = ' + assignment + ';' + "\n"
        return func

    def _get_fixed_parameters_as_var_data(self):
        var_data_dict = {}
        for m, p in self._model_functions_info.get_free_parameters_list():
            value = self._parameter_values['{}.{}'.format(m.name, p.name)]

            if self._model_functions_info.is_fixed_to_value('{}.{}'.format(m.name, p.name)) \
                and not all_elements_equal(value) \
                and not self._model_functions_info.is_parameter_fixed_to_dependency(m, p):

                if self.problems_to_analyze is not None:
                    value = value[self.problems_to_analyze, ...]

                var_data_dict.update({m.name + '_' + p.name: SimpleDataAdapter(value, p.data_type,
                                                                               self._get_mot_float_type())})
        return var_data_dict

    def _get_static_parameters_as_var_data(self):
        static_data_dict = {}

        for m, p in self._model_functions_info.get_static_parameters_list():
            static_map_value = self._get_static_map_value(m, p)

            if not all_elements_equal(static_map_value):
                data_adapter = SimpleDataAdapter(static_map_value, p.data_type, self._get_mot_float_type())
                static_data_dict.update({m.name + '_' + p.name: data_adapter})

        return static_data_dict

    def _get_bounds_as_var_data(self):
        bounds_dict = {}

        for m, p in self._model_functions_info.get_free_parameters_list():
            lower_bound = self._lower_bounds['{}.{}'.format(m.name, p.name)]
            upper_bound = self._upper_bounds['{}.{}'.format(m.name, p.name)]

            if not all_elements_equal(lower_bound):
                data_adapter = SimpleDataAdapter(lower_bound, p.data_type, self._get_mot_float_type())
                bounds_dict.update({'lb_' + m.name + '_' + p.name: data_adapter})

            if not all_elements_equal(upper_bound):
                data_adapter = SimpleDataAdapter(upper_bound, p.data_type, self._get_mot_float_type())
                bounds_dict.update({'ub_' + m.name + '_' + p.name: data_adapter})

        return bounds_dict

    def _get_static_map_value(self, model, parameter):
        """Get the map value for the given parameter of the given model.

        This first checks if the parameter is defined in the static maps data in the problem data. If not, we try
        to get it from the value stored in the parameter itself. If that fails as well we raise an error.

        Also, this only returns the problems for which problems_to_analyze is set.

        Args:
            model (ModelFunction): the model function
            parameter (CLParameter): the parameter for which we want to get the value

        Returns:
            ndarray or number: the value for the given parameter.
        """
        data = None
        value = self._parameter_values.get('{}.{}'.format(model.name, parameter.name), None)
        if parameter.name in self._problem_data.static_maps:
            data = self._problem_data.static_maps[parameter.name]
        elif value is not None:
            data = value

        if data is None:
            raise ValueError('No suitable data could be found for the static parameter {}.'.format(parameter.name))

        if is_scalar(data):
            return data

        if self.problems_to_analyze is not None:
            return data[self.problems_to_analyze, ...]
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
        name = m.name + '_' + p.name
        assignment = ''

        if isinstance(p, ProtocolParameter):
            assignment = 'data->protocol_data_' + p.name + '[observation_index]'
        elif isinstance(p, FreeParameter):
            value = self._parameter_values['{}.{}'.format(m.name, p.name)]

            if self._model_functions_info.is_fixed_to_value('{}.{}'.format(m.name, p.name)) \
                and not self._model_functions_info.parameter_has_dependency(m, p):
                if all_elements_equal(value):
                    assignment = '(' + data_type + ')' + str(float(get_single_value(value)))
                else:
                    assignment = '(' + data_type + ') data->var_data_' + m.name + '_' + p.name
            elif not self._model_functions_info.parameter_has_dependency(m, p) \
                or (self._model_functions_info.parameter_has_dependency(m, p)
                    and not self._model_functions_info.is_parameter_fixed_to_dependency(m, p)):
                ind = self._model_functions_info.get_parameter_estimable_index(m, p)
                assignment += 'x[' + str(ind) + ']'
            if self._model_functions_info.parameter_has_dependency(m, p):
                return self._get_dependent_parameters_listing(((m, p),))

        return data_type + ' ' + name + ' = ' + assignment + ';' + "\n"

    def _get_parameter_type_lists(self):
        """Returns a dictionary with the parameters sorted in the types protocol, fixed, estimable and dependent.

        Parameters may occur in different lists (estimable and dependent for example).
        """
        protocol_parameters = self._model_functions_info.get_protocol_parameters_list()
        fixed_parameters = []
        estimable_parameters = []
        depended_parameters = []

        for m, p in self._model_functions_info.get_free_parameters_list(exclude_priors=True):
            if self._model_functions_info.is_fixed_to_value('{}.{}'.format(m.name, p.name)) \
                and not self._model_functions_info.parameter_has_dependency(m, p):
                fixed_parameters.append((m, p))

            elif self._model_functions_info.is_parameter_estimable(m, p):
                estimable_parameters.append((m, p))

            if self._model_functions_info.parameter_has_dependency(m, p):
                ind = self._dependency_store.get_index(m.name + '.' + p.name)
                depended_parameters.insert(ind, (m, p))

        return {'protocol': protocol_parameters, 'fixed': fixed_parameters,
                'estimable': estimable_parameters, 'dependent': depended_parameters}

    def _convert_parameters_dot_to_bar(self, string):
        """Convert a string containing parameters with . to parameter names with _"""
        for dname, bname in self._parameters_dot_to_bar.items():
            string = string.replace(dname, bname)
        return string

    def _init_fixed_duplicates_dependencies(self):
        """Find duplicate fixed parameters, and make dependencies of them. This saves data transfer in CL."""
        var_data_dict = {}
        for m, p in self._model_functions_info.get_free_parameters_list():
            value = self._parameter_values['{}.{}'.format(m.name, p.name)]

            if self._model_functions_info.is_fixed_to_value('{}.{}'.format(m.name, p.name)) \
                    and not is_scalar(value) \
                    and not self._model_functions_info.is_parameter_fixed_to_dependency(m, p):

                duplicate_found = False
                duplicate_key = None

                for key, data in var_data_dict.items():
                    if np.array_equal(data, value):
                        duplicate_found = True
                        duplicate_key = key
                        break

                if duplicate_found:
                    self.add_parameter_dependency(m.name + '.' + p.name, SimpleAssignment(duplicate_key))
                else:
                    var_data_dict.update({m.name + '.' + p.name: value})

    def _get_variable_data(self):
        """See super class OptimizeModelInterface for details

        When overriding this function, please note that it should adhere to the attribute problems_to_analyze.
        """
        var_data_dict = {}

        observations = self._problem_data.observations
        if observations is not None:
            if self.problems_to_analyze is not None:
                observations = observations[self.problems_to_analyze, ...]

            observations = self._transform_observations(observations)

            data_adapter = SimpleDataAdapter(observations, CLDataType.from_string('mot_float_type*'),
                                             self._get_mot_float_type())
            var_data_dict.update({'observations': data_adapter})

        var_data_dict.update(self._get_fixed_parameters_as_var_data())
        var_data_dict.update(self._get_static_parameters_as_var_data())
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
                    exception = 'Protocol parameter "{}" could not be resolved'.format(m.name + '.' + p.name)
                    raise ParameterResolutionException(exception)
        return return_data

    def _get_static_data(self):
        model_data_dict = {}
        for m, p in self._model_functions_info.get_model_parameter_list():
            if isinstance(p, ModelDataParameter):
                value = self._parameter_values['{}.{}'.format(m.name, p.name)]
                if not all_elements_equal(value):
                    model_data_dict.update({p.name: SimpleDataAdapter(value, p.data_type, self._get_mot_float_type())})
        return model_data_dict

    def _get_all_kernel_source_items(self, device, problem_id_name='gid'):
        """Get the CL strings for the kernel source items for most common CL kernels in this library."""
        import pyopencl as cl

        max_constant_buffer_size = device.get_info(cl.device_info.MAX_CONSTANT_BUFFER_SIZE)
        max_constant_args = device.get_info(cl.device_info.MAX_CONSTANT_ARGS)

        def _check_array_fits_constant_buffer(array, dtype):
            """Check if the given array when casted to the given type can be fit into the given max_size

            Args:
                array (ndarray): the array we want to fit
                dtype (np data type): the numpy data type we want to use

            Returns:
                boolean: if it fits in the constant memory buffer or not
            """
            return np.product(array.shape) * np.dtype(dtype).itemsize < max_constant_buffer_size

        constant_args_counter = 0

        kernel_param_names = []
        data_struct_init = []
        data_struct_names = []

        for key, data_adapter in self._get_variable_data().items():
            clmemtype = 'global'

            cl_data = data_adapter.get_opencl_data()

            if _check_array_fits_constant_buffer(cl_data, data_adapter.get_opencl_numpy_type()):
                if constant_args_counter < max_constant_args:
                    clmemtype = 'constant'
                    constant_args_counter += 1

            param_name = 'var_data_' + str(key)
            data_type = data_adapter.get_data_type().raw_data_type

            if data_adapter.get_data_type().is_vector_type:
                data_type += data_adapter.get_data_type().vector_length

            kernel_param_names.append(clmemtype + ' ' + data_type + '* ' + param_name)

            mult = cl_data.shape[1] if len(cl_data.shape) > 1 else 1
            if len(cl_data.shape) == 1 or cl_data.shape[1] == 1:
                data_struct_names.append(data_type + ' ' + param_name)
                data_struct_init.append(param_name + '[{} * {}]'.format(problem_id_name, mult))
            else:
                data_struct_names.append(clmemtype + ' ' + data_type + '* ' + param_name)
                data_struct_init.append(param_name + ' + {} * {}'.format(problem_id_name, mult))

        for key, data_adapter in self._get_protocol_data().items():
            clmemtype = 'global'

            cl_data = data_adapter.get_opencl_data()

            if _check_array_fits_constant_buffer(cl_data, data_adapter.get_opencl_numpy_type()):
                if constant_args_counter < max_constant_args:
                    clmemtype = 'constant'
                    constant_args_counter += 1

            param_name = 'protocol_data_' + str(key)
            data_type = data_adapter.get_data_type().raw_data_type

            if data_adapter.get_data_type().is_vector_type:
                data_type += str(data_adapter.get_data_type().vector_length)

            kernel_param_names.append(clmemtype + ' ' + data_type + '* ' + param_name)
            data_struct_init.append(param_name)
            data_struct_names.append(clmemtype + ' ' + data_type + '* ' + param_name)

        for key, data_adapter in self._get_static_data().items():
            clmemtype = 'global'
            param_name = 'model_data_' + str(key)
            data_type = data_adapter.get_data_type().raw_data_type

            if data_adapter.get_data_type().is_vector_type:
                data_type += data_adapter.get_data_type().vector_length

            data_struct_init.append(param_name)

            if isinstance(data_adapter.get_opencl_data(), np.ndarray):
                kernel_param_names.append(clmemtype + ' ' + data_type + '* ' + param_name)
                data_struct_names.append(clmemtype + ' ' + data_type + '* ' + param_name)
            else:
                kernel_param_names.append(data_type + ' ' + param_name)
                data_struct_names.append(data_type + ' ' + param_name)

        data_struct = '''
            typedef struct{
                ''' + ('' if data_struct_names else 'constant void* place_holder;') + '''
                ''' + " ".join((name + ";\n" for name in data_struct_names)) + '''
            } ''' + self.get_kernel_data_struct_type() + ''';
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
                self.add_parameter_dependency(names[0], SimpleAssignment('1 - ({})'.format(' + '.join(names[1:]))))

    def _get_mot_float_type(self):
        """Get the data type for the mot_float_type"""
        if self.double_precision:
            return CLDataType.from_string('double')
        return CLDataType.from_string('float')

    def _get_weight_sum_to_one_transformation(self):
        """Returns a snippit of CL for the encode and decode functions to force the sum of the weights to 1"""
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


class SampleModelBuilder(OptimizeModelBuilder, SampleModelInterface):

    def __init__(self, model_name, model_tree, evaluation_model, signal_noise_model=None, problem_data=None,
                 enforce_weights_sum_to_one=True):
        super(SampleModelBuilder, self).__init__(model_name, model_tree, evaluation_model, signal_noise_model,
                                                 problem_data, enforce_weights_sum_to_one=enforce_weights_sum_to_one)

    def _init_model_information_container(self, dependency_store, model_tree, evaluation_model, signal_noise_model):
        """Get the model information container object.

        This is called in the __init__ to provide the new model with the correct subclass function information
        object. The rationale is that some subclasses may have additional parameters not present in optimization. For
        example, in sampling one can have priors with parameters. These parameters must be added to the model and the
        best point to do that is in the ModelFunctionsInformation object.

        Returns:
            ModelFunctionsInformation: the model function information object
        """
        return ModelFunctionsInformation(dependency_store, model_tree, evaluation_model, signal_noise_model,
                                         enable_prior_parameters=True)

    def get_log_prior_function(self, func_name='getLogPrior', address_space_parameter_vector='private'):
        prior = ''

        for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
            prior += p.sampling_prior.get_prior_function()

        prior += '''
            mot_float_type {func_name}(const void* data_void,
                                       {address_space_parameter_vector} const mot_float_type* const x){{

                {kernel_data_struct_type}* data = ({kernel_data_struct_type}*)data_void;
                mot_float_type prior = 1.0;

            '''.format(func_name=func_name, address_space_parameter_vector=address_space_parameter_vector,
                       kernel_data_struct_type=self.get_kernel_data_struct_type())

        for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
            name = m.name + '.' + p.name

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
                        value = self._parameter_values['{}.{}'.format(m.name, prior_param.name)]
                        if all_elements_equal(value):
                            prior_params.append(str(get_single_value(value)))
                        else:
                            prior_params.append('data->var_data_' + '{}_{}'.format(m.name, prior_param.name))

                prior += '\tprior *= {}(x[{}], {}, {}, {});\n'.format(function_name, i, lower_bound, upper_bound,
                                                                      ', '.join(prior_params))
            else:
                prior += '\tprior *= {}(x[{}], {}, {});\n'.format(function_name, i, lower_bound, upper_bound)

        weight_indices = []
        for (m, p) in self._model_functions_info.get_estimable_weights():
            weight_indices.append(self._model_functions_info.get_parameter_estimable_index(m, p))

        if weight_indices:
            prior += '''
                mot_float_type _weight_sum = ''' + ' + '.join('x[{}]'.format(index) for index in weight_indices) + ''';
                if(_weight_sum > 1.0){
                    prior *= 0;
                }
            '''

        prior += '\n\treturn log(prior);\n}'
        return prior

    def get_proposal_state(self):
        return_list = []
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            for param in p.sampling_proposal.get_parameters():
                if param.adaptable:
                    return_list.append(param.default_value)
        return return_list

    def get_proposal_state_names(self):
        return_list = []
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            for param in p.sampling_proposal.get_parameters():
                if param.adaptable:
                    return_list.append('{}.{}.proposal.{}'.format(m.name, p.name, param.name))
        return return_list

    def is_proposal_symmetric(self):
        return all(p.sampling_proposal.is_symmetric() for m, p in
                   self._model_functions_info.get_estimable_parameters_list())

    def get_proposal_logpdf(self, func_name='getProposalLogPDF', address_space_proposal_state='private'):
        return_str = ''
        for _, p in self._model_functions_info.get_estimable_parameters_list():
            return_str += p.sampling_proposal.get_proposal_logpdf_function()

        return_str += '''
            mot_float_type {func_name}(
                const int i,
                const mot_float_type proposal,
                const mot_float_type current,
                {address_space_proposal_state} mot_float_type* const proposal_state){{

                switch(i){{
        '''.format(func_name=func_name, address_space_proposal_state=address_space_proposal_state)

        adaptable_parameter_count = 0
        for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
            return_str += 'case ' + str(i) + ':' + "\n\t\t\t"

            param_proposal = p.sampling_proposal
            logpdf_call = 'return ' + param_proposal.get_proposal_logpdf_function_name() + '(proposal, current'

            for param in param_proposal.get_parameters():
                if param.adaptable:
                    logpdf_call += ', proposal_state[' + str(adaptable_parameter_count) + ']'
                    adaptable_parameter_count += 1
                else:
                    logpdf_call += ', ' + str(param.default_value)

            logpdf_call += ');'
            return_str += logpdf_call + "\n"

        return_str += "\n\t\t" + '}' + "\n" + 'return 0;' + "\n"
        return_str += '}'
        return return_str

    def get_proposal_function(self, func_name='getProposal', address_space_proposal_state='private'):
        return_str = ''
        for _, p in self._model_functions_info.get_estimable_parameters_list():
            return_str += p.sampling_proposal.get_proposal_function()

        return_str += '''
            mot_float_type {func_name}(
                const int i,
                const mot_float_type current,
                void* rng_data,
                {address_space_proposal_state} mot_float_type* const proposal_state){{

                switch(i){{
        '''.format(func_name=func_name, address_space_proposal_state=address_space_proposal_state)

        adaptable_parameter_count = 0
        for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
            return_str += 'case ' + str(i) + ':' + "\n\t\t\t"

            param_proposal = p.sampling_proposal
            proposal_call = 'return ' + param_proposal.get_proposal_function_name() + '(current, rng_data'

            for param in param_proposal.get_parameters():
                if param.adaptable:
                    proposal_call += ', proposal_state[' + str(adaptable_parameter_count) + ']'
                    adaptable_parameter_count += 1
                else:
                    proposal_call += ', ' + str(param.default_value)

            proposal_call += ');'
            return_str += proposal_call + "\n"

        return_str += "\n\t\t" + '}' + "\n" + 'return 0;' + "\n"
        return_str += '}'
        return return_str

    def get_proposal_state_update_function(self, func_name='updateProposalState', address_space='private'):
        return_str = ''
        for _, p in self._model_functions_info.get_estimable_parameters_list():
            if p.sampling_proposal.is_adaptable():
                return_str += p.sampling_proposal.get_proposal_update_function().get_update_function(
                    p.sampling_proposal.get_parameters(), address_space=address_space)

        return_str += '''
            void {func_name}({address_space} mot_float_type* const proposal_state,
                             {address_space} uint* const sampling_counter,
                             {address_space} uint* const acceptance_counter,
                             {address_space} mot_float_type* const parameter_variance){{
        '''.format(func_name=func_name, address_space=address_space)

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

                return_str += '''
                    // {param_name}
                    {update_func_name}({params});
                '''.format(update_func_name=proposal_update_function.get_function_name(param_proposal.get_parameters()),
                           params=', '.join(state_params), param_name='{}.{}'.format(m.name, p.name))

        return_str += '}'
        return return_str

    def get_log_likelihood_function(self, func_name='getLogLikelihood', evaluation_model=None, full_likelihood=True):
        evaluation_model = evaluation_model or self._evaluation_model

        inst_per_problem = self.get_nmr_inst_per_problem()
        eval_func_name = func_name + '_evaluateModel'
        obs_func_name = func_name + '_getObservation'

        param_listing = ''
        for p in evaluation_model.get_free_parameters():
            param_listing += self._get_param_listing_for_param(evaluation_model, p)

        func = ''
        func += evaluation_model.get_cl_dependency_headers()
        func += evaluation_model.get_cl_dependency_code()

        func += self.get_model_eval_function(eval_func_name)
        func += self.get_observation_return_function(obs_func_name)
        func += evaluation_model.get_log_likelihood_function(func_name, inst_per_problem, eval_func_name,
                                                             obs_func_name, param_listing,
                                                             full_likelihood=full_likelihood)
        return func

    def get_log_likelihood_per_observation_function(self, func_name="getLogLikelihoodPerObservation",
                                                    evaluation_model=None, full_likelihood=True):
        evaluation_model = evaluation_model or self._evaluation_model

        inst_per_problem = self.get_nmr_inst_per_problem()
        eval_func_name = func_name + '_evaluateModel'
        obs_func_name = func_name + '_getObservation'

        param_listing = ''
        for p in evaluation_model.get_free_parameters():
            param_listing += self._get_param_listing_for_param(evaluation_model, p)

        func = ''
        func += evaluation_model.get_cl_dependency_headers()
        func += evaluation_model.get_cl_dependency_code()

        func += self.get_model_eval_function(eval_func_name)
        func += self.get_observation_return_function(obs_func_name)
        func += evaluation_model.get_log_likelihood_per_observation_function(
            func_name, inst_per_problem, eval_func_name,
            obs_func_name, param_listing,
            full_likelihood=full_likelihood)
        return func

    def samples_to_statistics(self, samples_dict):
        results = {}
        for key, value in samples_dict.items():
            _, param = self._model_functions_info.get_model_parameter_by_name(key)
            stat_mod = param.sampling_statistics
            results[key] = stat_mod.get_mean(value)
            results[key + '.std'] = stat_mod.get_std(value)
        return results


class _DependencyStore(object):

    def __init__(self):
        self.names_in_order = []
        self.dependencies = {}

    def set_dependency(self, param_name, dependency):
        if param_name not in self.names_in_order:
            self.names_in_order.append(param_name)
        self.dependencies.update({param_name: dependency})

    def get_dependency(self, param_name):
        return self.dependencies[param_name]

    def has_dependency(self, param_name):
        return param_name in self.dependencies

    def has_dependencies(self):
        return self.names_in_order

    def get_index(self, param_name):
        return self.names_in_order.index(param_name)


class ModelFunctionsInformation(object):

    def __init__(self, dependency_store, model_tree, evaluation_model, signal_noise_model=None,
                 enable_prior_parameters=False):
        """Contains centralized information about the model functions in the model builder parent.

        Args:
            dependency_store (_DependencyStore): the shared dependency store containing information about
                the parameters that depend on each other
            model_tree (mot.model_building.trees.CompartmentModelTree): the model tree object
            evaluation_model (mot.model_building.evaluation_models.EvaluationModel): the evaluation model to
                use for the resulting complete model
            signal_noise_model (mot.model_building.signal_noise_models.SignalNoiseModel): the signal
                noise model to use to add noise to the model prediction
            enable_prior_parameters (boolean): adds possible prior parameters to the list of parameters in the model
        """
        self._dependency_store = dependency_store
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

    def set_fixed_to_value(self, parameter_name, fix_state):
        """Set the given parameter fixed.

        This only works with free parameters.

        Args:
            parameter_name (str): the name of the parameter to fix or unfix
            fix_state (boolean): if the parameter is fixed or not
        """
        self._fixed_parameters[parameter_name] = fix_state

    def is_fixed_to_value(self, parameter_name):
        """Check if the given (free) parameter is fixed to a value.

        Args:
            parameter_name (str): the name of the parameter to fix or unfix

        Returns:
            boolean: if the parameter is fixed to a value or not
        """
        return self._fixed_parameters[parameter_name]

    def get_model_list(self):
        """Get the list of all the applicable model functions

        Returns:
            list of mot.model_building.cl_functions.base.ModelFunction: the list of model functions.
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
            exclude_priors (boolean): if we want to exclude the priors or not

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

    def get_non_model_tree_param_listing(self):
        """Get the model, parameter tuples for all parameters not in the model tree.

        Basically this returns the parameters of the evaluation and signal noise model.

        Returns:
            tuple: the (model, parameter) tuple for all non model tree parameters
        """
        listing = []
        for p in self._evaluation_model.parameter_list:
            listing.append((self._evaluation_model, p))

        if self._signal_noise_model:
            for p in self._signal_noise_model.parameter_list:
                listing.append((self._signal_noise_model, p))

        return listing

    def parameter_has_dependency(self, model, param):
        """Check if the given model and parameter name combo has a dependency.

        Args:
            model (mot.model_building.cl_functions.base.ModelFunction): the model function
            param (mot.model_building.cl_functions.parameters.CLFunctionParameter): the parameter

        Returns:
            boolean: if the given parameter has a dependency
        """
        return self._dependency_store.has_dependency(model.name + '.' + param.name)

    def is_parameter_fixed_to_dependency(self, model, param):
        """Check if the given model and parameter name combo has a dependency.

        Args:
            model (mot.model_building.cl_functions.base.ModelFunction): the model function
            param (mot.model_building.cl_functions.parameters.CLFunctionParameter): the parameter

        Returns:
            boolean: if the given parameter is fixed to a dependency. Returns False if either this parameter
                has no dependency or if it is not fixed to it.
        """
        return self.parameter_has_dependency(model, param)

    def is_parameter_estimable(self, model, param):
        """Check if the given model parameter is estimable.

        A parameter is estimable if it is free, not fixed to dependencies and not fixed to any static value.

        Args:
            model (mot.model_building.cl_functions.base.ModelFunction): the model function
            param (mot.model_building.cl_functions.parameters.CLFunctionParameter): the parameter

        Returns:
            boolean: true if the parameter is estimable, false otherwise
        """
        return isinstance(param, FreeParameter) and \
            not self._fixed_parameters.get('{}.{}'.format(model.name, param.name), False) and \
            not self.is_parameter_fixed_to_dependency(model, param)

    def get_estimable_parameters_list(self):
        """Gets a list (as model, parameter tuples) of all parameters that are estimable.

        Returns:
            list of tuple: the list of estimable parameters
        """
        estimable_parameters = [(m, p) for m, p in self._model_parameter_list if self.is_parameter_estimable(m, p)]

        if self._enable_prior_parameters:
            prior_params = []
            for m, p in estimable_parameters:
                prior_params.extend((m, prior_p) for prior_p in m.get_prior_parameters(p) if not prior_p.fixed)
            estimable_parameters.extend(prior_params)

        return estimable_parameters

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
        return list((m, p) for m in self._model_list for p in m.parameter_list)

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
            model (mot.model_building.cl_functions.base.ModelFunction): the model function
            param (mot.model_building.cl_functions.parameters.CLFunctionParameter): the parameter

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


class ParameterNameException(Exception):
    """Thrown when the a parameter of an given name could not be found."""
    pass


class ParameterResolutionException(Exception):
    """Thrown when a fixed parameter could not be resolved."""
    pass


class DoubleModelNameException(Exception):
    """Thrown when there are two models with the same name."""
    pass

