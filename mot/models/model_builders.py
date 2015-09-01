import numbers
import numpy as np
from ..base import ProtocolParameter, ModelDataParameter, FreeParameter, CLDataType
from .. import runtime_configuration
from ..cl_routines.mapping.calc_dependent_params import CalculateDependentParameters
from ..utils import set_cl_compatible_data_type, TopologicalSort, set_correct_cl_data_type
from ..parameter_functions.codecs import CodecBuilder
from ..parameter_functions.dependencies import SimpleAssignment
from ..models.interfaces import OptimizeModelInterface, SampleModelInterface


__author__ = 'Robbert Harms'
__date__ = "2014-03-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class OptimizeModelBuilder(OptimizeModelInterface):

    def __init__(self, name, model_tree, evaluation_model, signal_noise_model=None, problem_data=None):
        """Create a new model builder that can construct an optimization model using parts.

        Args:
            name (str): the name of the model
            model_tree (CompartmentModelTree): the model tree object
            evaluation_model (EvaluationModel): the evaluation model to use for the resulting complete model
            signal_noise_model (SignalNoiseModel): the optional signal noise model to use to noise the model prediction
            problem_data (ProblemData): the problem data object
        """
        super(OptimizeModelBuilder, self).__init__()
        self._name = name
        self._double_precision = False
        self._model_tree = model_tree
        self._evaluation_model = evaluation_model
        self._signal_noise_model = signal_noise_model
        self._parameters_dot_to_bar = {}
        self._dependency_store = DependencyStore()
        self._post_optimization_modifiers = []

        self._problem_data = None
        if problem_data:
            self.set_problem_data(problem_data)

        for m, p in self._get_model_parameter_list():
            self._parameters_dot_to_bar.update({m.name + '.' + p.name: m.name + '_' + p.name})

        self._check_for_double_model_names()

        self._set_default_dependencies()

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
            value: boolean: if we would like to do the computations in double of single floating point type.
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
        m, p = self._get_model_parameter_matching(model_param_name)
        p.parameter_state.fix_to(value)
        return self

    def init(self, model_param_name, value):
        """Init the given model.param to the given value.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'
            value (scalar or vector): The value to initialize the given parameter to

        Returns:
            Returns self for chainability
        """
        m, p = self._get_model_parameter_matching(model_param_name)
        p.parameter_state.value = value
        return self

    def unfix(self, model_param_name):
        """Unfix the given model.param

        Args:
            model_param_name (string):
                A model.param name like 'Ball.d'

        Returns:
            Returns self for chainability
        """
        m, p = self._get_model_parameter_matching(model_param_name)
        p.parameter_state.fixed = False
        return self

    def set_problem_data(self, problem_data):
        """Set the data this model will deal with. This overwrites those optionally set in the constructor.

        Args:
            problem_data (ProblemData):
                The container for the problem data we will use for this model.

        Returns:
            Returns self for chainability
        """
        self._problem_data = problem_data
        return self

    def cmf(self, model_name):
        """Get the Compartment Model Function object corresponding to the given model name.

        This may be useful for later fixing or adding items to a specific model.

        Returns:
            CLFunction: the compartment model function with the given name. None if no matching function found.
        """
        models = self._get_model_list()
        for m in models:
            if m.name == model_name:
                return m
        return None

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
        """
        self._post_optimization_modifiers.append((model_param_name, mod_routine))
        return self

    def add_post_optimization_modifiers(self, modifiers):
        """Add a list of modifier functions.

        The same as add_post_optimization_modifier() except that it accepts a list of lists. Every element in the list
        should be a tuple like (model_param_name, mod_routine)

        Args:
            modifiers (tuple of tuples): The list of modifiers to add (in order).

        """
        self._post_optimization_modifiers.extend(modifiers)

    def get_required_protocol_names(self):
        """Get a list with the constant data names that are needed for this model to work.

        For example, an implementing diffusion MRI model might require the presence of the protocol parameter
        'g' and 'b'. This function should then return ('g', 'b').

        Returns:
            A list of columns names that are to be taken from the protocol data.
        """
        return list(set([p.name for m, p in self._get_model_parameter_list() if
                         isinstance(p, ProtocolParameter)]))

    def get_problems_var_data(self):
        """See super class OptimizeModelInterface for details"""
        var_data_dict = {'observations': set_cl_compatible_data_type(
            self._problem_data.observations, CLDataType.from_string('model_float*'), self._double_precision)}
        var_data_dict.update(self._get_fixed_parameters_as_var_data())
        return var_data_dict

    def get_optimization_output_param_names(self):
        """See super class OptimizeModelInterface for details"""
        items = []
        for m in self._get_model_list():
            for p in m.get_free_parameters():
                items.append(m.name + '.' + p.name)

        for name, _ in self._post_optimization_modifiers:
            items.append(name)
        return items

    def get_optimized_param_names(self):
        return [m.name + '.' + p.name for m, p in self._get_estimable_parameters_list()]

    def get_nmr_problems(self):
        return self._problem_data.observations.shape[0]

    def get_nmr_inst_per_problem(self):
        return self._problem_data.observations.shape[1]

    def get_nmr_estimable_parameters(self):
        return len(self.get_optimized_param_names())

    def get_problems_prtcl_data(self):
        prtcl_data_dict = {}
        for m, p in self._get_model_parameter_list():
            if isinstance(p, ProtocolParameter):
                if p.name in self._problem_data.prtcl_data_dict:
                    if not self._all_elements_equal(self._problem_data.prtcl_data_dict[p.name]):
                        const_d = {p.name: set_cl_compatible_data_type(self._problem_data.prtcl_data_dict[p.name],
                                                                       p.cl_data_type, self.double_precision)}
                        prtcl_data_dict.update(const_d)
                else:
                    exception = 'Constant parameter "{}" could not be resolved'.format(m.name + '.' + p.name)
                    raise ParameterResolutionException(exception)
        return prtcl_data_dict

    def get_problems_fixed_data(self):
        fixed_data_dict = {}
        for m, p in self._get_model_parameter_list():
            if isinstance(p, ModelDataParameter):
                fixed_data_dict.update({p.name: set_cl_compatible_data_type(p.value, p.cl_data_type, self.double_precision)})
        return fixed_data_dict

    def get_initial_parameters(self, results_dict=None):
        starting_points = []
        for m, p in self._get_estimable_parameters_list():
            if results_dict and (m.name + '.' + p.name) in results_dict:
                starting_points.append(results_dict[m.name + '.' + p.name])
            elif isinstance(p.value, numbers.Number):
                starting_points.append(np.tile(np.array(p.value), (self.get_nmr_problems(), 1)))
            else:
                if len(p.value.shape) < 2:
                    starting_points.append(np.transpose(np.asarray([p.value])))
                elif p.value.shape[1] > p.value.shape[0]:
                    starting_points.append(np.transpose(p.value))
                else:
                    starting_points.append(p.value)

        starting_points = [np.transpose(np.array([s])) if len(s.shape) < 2 else s for s in starting_points]
        #todo make model_float
        return set_correct_cl_data_type(np.concatenate(starting_points, axis=1))

    def get_lower_bounds(self):
        return np.array([p.lower_bound for m, p in self._get_estimable_parameters_list()])

    def get_upper_bounds(self):
        return np.array([p.upper_bound for m, p in self._get_estimable_parameters_list()])

    def set_initial_parameters(self, initial_params):
        """Update the initial parameters for this model by the given values. This only affects free
        and non fixed parameters.

        Args:
            - initial_params: a dictionary containing as keys full parameter names (<model>_<param>) and as values
                numbers or arrays to be used as starting point
        """
        for m, p in self._get_estimable_parameters_list():
            if m.name + '.' + p.name in initial_params:
                p.value = initial_params[m.name + '.' + p.name]
        return self

    def get_parameter_codec(self):
        dep_list = {}
        transform_dict = {}
        for m, p in self._get_estimable_parameters_list():
            name = m.name + '.' + p.name
            dep_names = list(dep[0].name + '.' + dep[1].name for dep in p.parameter_transform.dependencies)
            dep_list.update({name: dep_names})
            transform_dict.update({name: p})

        dep_list = TopologicalSort(dep_list).get_flattened_sort()

        dec_func_list = []
        enc_func_list = []
        for name in dep_list:
            parameter = transform_dict[name]
            ind = self._get_parameter_estimable_index(name)
            transform = parameter.parameter_transform

            deps_names = []
            for dep in transform.dependencies:
                dep_ind = self._get_parameter_estimable_index(dep[0].name + '.' + dep[1].name)
                deps_names.append('{0}[' + str(dep_ind) + ']')

            s = '{0}[' + str(ind) + '] = ' + transform.get_cl_decode(parameter, '{0}[' + str(ind) + ']', deps_names)
            dec_func_list.append(s)

            s = '{0}[' + str(ind) + '] = ' + transform.get_cl_encode(parameter, '{0}[' + str(ind) + ']', deps_names)
            enc_func_list.append(s)

        return CodecBuilder(list(reversed(enc_func_list)), dec_func_list)

    def get_final_parameter_transformations(self, func_name='applyFinalParameterTransformations'):
        """Get the transformations that must be applied at the end of an optimization (or sampling) routine.

        These transformations must contain all parameter dependencies, as such that all transformation happening in the
        model function which do not happen in the codec must also go here.

        Returns:
            str: A function of the kind: void finalParameterTransformations(const optimize_data* data, double* x)
                Which is called for every voxel and must in place edit the x variable.
        """
        transform_needed = any(dp.has_side_effects or not dp.fixed for dp in
                               self._dependency_store.dependencies.values())

        if not self._dependency_store.has_dependencies() or not transform_needed:
            return None

        param_exclude_list = [m.name + '_' + p.name for (m, p) in self._get_non_model_tree_param_listing()]
        param_lists = self._get_parameter_type_lists()
        estimable_names = [m.name + '_' + p.name for m, p in param_lists['estimable']]
        depend_param_listing = self._get_dependent_parameters_listing(param_lists['dependent'])

        for m, p in param_lists['fixed'] + param_lists['constant']:
            if (m.name + '_' + p.name) not in depend_param_listing:
                param_exclude_list.append(m.name + '_' + p.name)

        for m, p in param_lists['dependent']:
            if self._parameter_fixed_to_dependency(m, p):
                if (m.name + '_' + p.name) not in estimable_names:
                    param_exclude_list.append(m.name + '_' + p.name)
        param_listing = self._get_parameters_listing(exclude_list=param_exclude_list)

        func = "\n\t\t\t" + 'void ' + func_name + '(const optimize_data* const data, model_float* const x){' + "\n"
        func += param_listing + "\n"

        for i, (m, p) in enumerate(self._get_parameter_type_lists()['estimable']):
            if not self._is_non_model_tree_model(m):
                func += "\t"*4 + 'x[' + str(i) + '] = ' + m.name + '_' + p.name + ';' + "\n"
        func += "\t\t\t" + '}' + "\n"
        return func

    def get_observation_return_function(self, func_name='getObservation'):
        func = '''
            model_float ''' + func_name + '''(const optimize_data* const data, const int observation_index){
                return data->var_data_observations[observation_index];
            }
        '''
        return func

    def get_model_eval_function(self, func_name='evaluateModel'):
        func = ''
        for leave in self._model_tree.leaves:
            func += leave.data.get_cl_header() + "\n"
            func += leave.data.get_cl_code() + "\n"

        noise_func_name = func_name + '_signalNoiseModel'
        if self._signal_noise_model:
            func += self._signal_noise_model.get_signal_function(noise_func_name)

        func += '''
            model_float ''' + func_name + \
                '(const optimize_data* const data, const double* const x, const int observation_index){' + "\n"
        func += self._get_parameters_listing(exclude_list=[m.name + '_' + p.name for (m, p) in
                                                           self._get_non_model_tree_param_listing()])

        if self._signal_noise_model:
            noise_params_listing = ''
            noise_params_func = ''
            for p in self._signal_noise_model.get_free_parameters():
                noise_params_listing += "\t" * 4 + self._get_param_listing_for_param(self._signal_noise_model, p)
                noise_params_func += ', ' + self._signal_noise_model.name + '_' + p.name
            func += "\n"
            func += noise_params_listing

            func += '''
                return ''' + noise_func_name + '''((''' + \
                self._build_model_from_tree(self._model_tree, 0) + ''')''' + noise_params_func +\
                ''');'''
        else:
            func += '''
                return (''' + self._build_model_from_tree(self._model_tree, 0) + ''');'''
        func += "\n\t\t\t}"
        return func

    def get_objective_function(self, func_name="calculateObjective"):
        inst_per_problem = self.get_nmr_inst_per_problem()
        eval_func_name = func_name + '_evaluateModel'
        obs_func_name = func_name + '_getObservation'

        param_listing = ''
        for p in self._evaluation_model.get_free_parameters():
            param_listing += self._get_param_listing_for_param(self._evaluation_model, p)

        func = self.get_model_eval_function(eval_func_name)
        func += self.get_observation_return_function(obs_func_name)
        func += self._evaluation_model.get_objective_function(func_name, inst_per_problem, eval_func_name,
                                                              obs_func_name, param_listing)
        return func

    def finalize_optimization_results(self, results_dict):
        """This adds the final optimization maps to the results dictionary.

        Steps in finalizing the results dict:
            1) It first adds the maps for the dependent and fixed parameters
            2) Second it adds the extra maps defined in the models itself.
            3) Finally it loops through the post_optimization_modifiers callback functions for the final updates.

        For more documentation see the base method.

        """
        self._add_dependent_parameter_maps(results_dict)
        self._add_fixed_parameter_maps(results_dict)

        for model in self._get_model_list():
            results_dict.update(model.get_extra_results_maps(results_dict))

        for name, routine in self._post_optimization_modifiers:
            results_dict[name] = routine(results_dict)

        return results_dict

    def _add_fixed_parameter_maps(self, results_dict):
        """In place add complete maps for the fixed parameters."""
        param_lists = self._get_parameter_type_lists()
        fixed_params = param_lists['fixed']
        for (m, p) in fixed_params:
            if not self._parameter_fixed_to_dependency(m, p):
                name = m.name + '.' + p.name
                if isinstance(p.value, numbers.Number):
                    results_dict.update({name: np.tile(np.array([p.value]), (self.get_nmr_problems(),))})
                else:
                    results_dict.update({name: p.value})

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

            cpd = CalculateDependentParameters(runtime_configuration.runtime_config['cl_environments'],
                                               runtime_configuration.runtime_config['load_balancer'], self.double_precision)
            dependent_parameters = cpd.calculate(self._get_fixed_parameters_as_var_data(),
                                                 estimated_parameters, func, dependent_parameter_names)
            results_dict.update(dependent_parameters)

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
                param_list.append('data->fixed_data_' + param.name)
            else:
                param_list.append(model.name + '_' + param.name)
        return model.cl_function_name + '(' + ', '.join(param_list) + ')'

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
        func += self._get_constant_parameters_listing(param_lists['constant'], exclude_list=exclude_list)
        func += self._get_fixed_parameters_listing(param_lists['fixed'], exclude_list=exclude_list)
        func += self._get_estimable_parameters_listing(param_lists['estimable'], exclude_list=exclude_list)
        func += self._get_dependent_parameters_listing(param_lists['dependent'], exclude_list=exclude_list)
        return func

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
                data_type = p.cl_data_type.cl_type
                assignment = 'x[' + str(estimable_param_counter) + ']'
                func += "\t"*4 + data_type + ' ' + name + ' = ' + assignment + ';' + "\n"
                estimable_param_counter += 1
        return func

    def _get_constant_parameters_listing(self, param_list=None, exclude_list=()):
        """Get the parameter listing for the constant parameters.

        For performance reasons, the parameter list should already be given.
            If not given it is calculated using:
                self._get_parameter_type_lists()['constant']

        Args:
            param_list: the list with the constant parameters
            exclude_list: a list of parameters to exclude from this listing
        """
        if param_list is None:
            param_list = self._get_parameter_type_lists()['constant']

        const_params_seen = []
        func = ''
        for m, p in param_list:
            if (m.name + '_' + p.name) not in exclude_list:
                data_type = p.cl_data_type.cl_type
                if p.name not in const_params_seen:
                    if self._all_elements_equal(self._problem_data.prtcl_data_dict[p.name]):
                        if p.cl_data_type.is_vector_type:
                            vector_length = p.cl_data_type.vector_length
                            values = [str(val) for val in self._problem_data.prtcl_data_dict[p.name][0]]
                            if len(values) < vector_length:
                                values.append(str(0))
                            assignment = '(' + data_type + ')(' + ', '.join(values) + ')'
                        else:
                            assignment = str(float(self._problem_data.prtcl_data_dict[p.name][0]))
                    else:
                        assignment = 'data->prtcl_data_' + p.name + '[observation_index]'
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
                data_type = p.cl_data_type.raw_data_type
                if isinstance(p.value, numbers.Number):
                    assignment = '(' + data_type + ')' + str(float(p.value))
                else:
                    if p.value.max() == p.value.min():
                        assignment = '(' + data_type + ')' + str(float(p.value[0]))
                    else:
                        assignment = 'data->var_data_' + m.name + '_' + p.name + '[0]'
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
            data_type = p.cl_data_type.raw_data_type

            if self._parameter_fixed_to_dependency(m, p):
                if (m.name + '_' + p.name) not in exclude_list:
                    func += "\t"*4 + data_type + ' ' + name + ' = ' + assignment + ';' + "\n"
            else:
                func += "\t"*4 + name + ' = ' + assignment + ';' + "\n"
        return func

    def _get_fixed_parameters_as_var_data(self):
        var_data_dict = {}
        for m, p in self._get_free_parameters_list():
            inlined_in_cl_code = isinstance(p.value, numbers.Number)

            if p.fixed and not inlined_in_cl_code and not self._parameter_fixed_to_dependency(m, p):
                var_data_dict.update({m.name + '_' + p.name: set_cl_compatible_data_type(p.value,
                                                                                         p.cl_data_type,
                                                                                         self.double_precision)})
        return var_data_dict

    def _get_non_model_tree_param_listing(self):
        listing = []
        for p in self._evaluation_model.parameter_list:
            listing.append((self._evaluation_model, p))

        if self._signal_noise_model:
            for p in self._signal_noise_model.parameter_list:
                listing.append((self._signal_noise_model, p))

        return listing

    def _is_non_model_tree_model(self, model):
        return model is self._evaluation_model or (self._signal_noise_model is not None and
                                                   model is self._signal_noise_model)

    def _get_param_listing_for_param(self, m, p):
        """Get the param listing for one specific parameter. This can be used for example for the noise model params.

        Please note, that on the moment this function does not support the complete dependency graph for the dependent
        parameters.
        """
        data_type = p.cl_data_type.raw_data_type
        name = m.name + '_' + p.name
        assignment = ''

        if isinstance(p, ProtocolParameter):
            assignment = 'data->prtcl_data_' + p.name + '[observation_index]'
        elif isinstance(p, FreeParameter):
            if p.fixed and not self._parameter_has_dependency(m, p):
                if isinstance(p.value, numbers.Number):
                    assignment = '(' + data_type + ')' + str(float(p.value))
                else:
                    if p.value.max() == p.value.min():
                        assignment = '(' + data_type + ')' + str(float(p.value[0]))
                    else:
                        assignment = 'data->var_data_' + m.name + '_' + p.name + '[0]'
            elif not self._parameter_has_dependency(m, p) or (self._parameter_has_dependency(m, p)
                                                              and not self._parameter_fixed_to_dependency(m, p)):
                ind = self._get_parameter_estimable_index(m.name + '.' + p.name)
                assignment += 'x[' + str(ind) + ']'
            if self._parameter_has_dependency(m, p):
                return self._get_dependent_parameters_listing(((m, p),))

        return data_type + ' ' + name + ' = ' + assignment + ';' + "\n"

    def _get_parameter_type_lists(self):
        """Returns a dictionary with the parameters sorted in the types constant, fixed, estimable and dependent.

        Parameters may occur in different lists (estimable and dependent for example).
        """
        constant_parameters = []
        fixed_parameters = []
        estimable_parameters = []
        depended_parameters = []

        for m, p in self._get_model_parameter_list():
            if isinstance(p, ProtocolParameter):
                constant_parameters.append((m, p))
            elif isinstance(p, FreeParameter):
                if p.fixed and not self._parameter_has_dependency(m, p):
                    fixed_parameters.append((m, p))
                elif not self._parameter_has_dependency(m, p) or (self._parameter_has_dependency(m, p)
                                                                  and not self._parameter_fixed_to_dependency(m, p)):
                    estimable_parameters.append((m, p))

                if self._parameter_has_dependency(m, p):
                    ind = self._dependency_store.get_index(m.name + '.' + p.name)
                    depended_parameters.insert(ind, (m, p))

        return {'constant': constant_parameters, 'fixed': fixed_parameters,
                'estimable': estimable_parameters, 'dependent': depended_parameters}

    def _get_parameter_by_name(self, parameter_name):
        """Get the parameter object of the given full parameter name in dot format.

        Args:
            parameter_name (string): the parameter name in dot format: <model>.<param>
        """
        models = self._get_model_list()
        for m in models:
            for p in m.parameter_list:
                if (m.name + '.' + p.name) == parameter_name:
                    return p

    def _get_parameter_estimable_index(self, parameter_name):
        """Get the index of this parameter in the parameters list (in CL x vector) of the given full parameter name.

        Args:
            parameter_name: the parameter name in dot format. <model>.<param>
        """
        estimable_param_counter = 0
        for m in self._get_model_list():
            for p in m.parameter_list:
                if (m.name + '.' + p.name) == parameter_name:
                    return estimable_param_counter

                if not isinstance(p, ProtocolParameter) \
                        and not isinstance(p, ModelDataParameter) \
                        and not self._parameter_fixed_to_dependency(m, p) \
                        and not p.fixed:
                    estimable_param_counter += 1

    def _get_model_list(self):
        """Get the list of all the Model Functions that play a role in this CompositeModel"""
        models = [n.data for n in self._model_tree.leaves]
        models.append(self._evaluation_model)
        if self._signal_noise_model:
            models.append(self._signal_noise_model)
        return models

    def _get_model_parameter_list(self):
        """Get a list of all model, parameter tuples."""
        return list((m, p) for m in self._get_model_list() for p in m.parameter_list)

    def _get_free_parameters_list(self):
        """Gets the free parameters (as model, parameter tuples) from the model listing.
        This does not incorporate checking for fixed parameters.
        """
        return list((m, p) for m in self._get_model_list() for p in m.get_free_parameters())

    def _get_estimable_parameters_list(self):
        """Gets a list (as model, parameter tuples) of all parameters that are estimable. """
        l = []
        for m in self._get_model_list():
            for p in m.get_free_parameters():
                if self._parameter_estimable(m, p):
                    l.append((m, p))
        return l

    def _get_model_estimable_parameters(self, model):
        return list(p for p in model.get_free_parameters() if self._parameter_estimable(model, p))

    def _parameter_estimable(self, m, p):
        return not p.fixed and not self._parameter_fixed_to_dependency(m, p)

    def _parameter_fixed_to_dependency(self, model, param):
        return self._parameter_has_dependency(model, param) \
            and self._dependency_store.get_dependency(model.name + '.' + param.name).fixed

    def _parameter_has_dependency(self, model, param):
        """Check if the given model and parameter name combo has a dependency."""
        return self._dependency_store.has_dependency(model.name + '.' + param.name)

    def _convert_parameters_dot_to_bar(self, string):
        """Convert a string containing parameters with . to parameter names with _"""
        for dname, bname in self._parameters_dot_to_bar.items():
            string = string.replace(dname, bname)
        return string

    def _init_fixed_duplicates_dependencies(self):
        """Find duplicate fixed parameters, and make dependencies of them. This saves data transfer in CL."""
        var_data_dict = {}
        for m, p in self._get_free_parameters_list():
            if p.fixed \
                    and not isinstance(p.value, numbers.Number) \
                    and not self._parameter_fixed_to_dependency(m, p):

                duplicate_found = False
                duplicate_key = None

                for key, data in var_data_dict.items():
                    if np.array_equal(data, p.value):
                        duplicate_found = True
                        duplicate_key = key
                        break

                if duplicate_found:
                    self.add_parameter_dependency(m.name + '.' + p.name, SimpleAssignment(duplicate_key))
                else:
                    var_data_dict.update({m.name + '.' + p.name: p.value})

    def _check_for_double_model_names(self):
        models = self._get_model_list()
        model_names = []
        for m in models:
            if m.name in model_names:
                raise DoubleModelNameException("Double model name detected in the model tree.", m.name)
            model_names.append(m.name)

    def _contains_parameter_reference(self, s):
        """Test if the given string contains a parameter reference to another model.
        Here s is a string given by the user for a dependency, transformation or other possible string items.

        Args:
            s: string
                The string we need to check for possible user referenced parameters

        Returns:
            Tuple with tuples for each (model, param) referenced in the given string
        """
        return list((m, p) for m, p in self._get_model_parameter_list() if m.name + '.' + p.name in s)

    def _get_model_parameter_matching(self, model_param_name):
        for m, p in self._get_model_parameter_list():
            if m.name + '.' + p.name == model_param_name:
                return m, p
        return None

    def _set_default_dependencies(self):
        """Initialize the default dependencies.

        By default this only adds dependencies for the fixed data that is used in multiple parameters.
        """
        self._init_fixed_duplicates_dependencies()

    def _all_elements_equal(self, col):
        """Checks if all elements of a given numpy column are equal to each other.

        Args:
            col (array): a numpy array

        Returns:
            bool: true if all elements are equal to each other, false otherwise
        """
        return (col == col[0]).all()


class SampleModelBuilder(OptimizeModelBuilder, SampleModelInterface):

    def __init__(self, model_name, model_tree, evaluation_model, signal_noise_model=None, problem_data=None):
        super(SampleModelBuilder, self).__init__(model_name, model_tree, evaluation_model, signal_noise_model,
                                                 problem_data)

    def get_log_prior_function(self, func_name='getLogPrior'):
        prior = 'double ' + func_name + '(const double* const x){' + "\n"
        prior += "\t" + 'double prior = 1.0;' + "\n"
        for i, (m, p) in enumerate(self._get_estimable_parameters_list()):
            prior += "\t" + 'prior *= ' + p.sampling_prior.get_cl_assignment(p, 'x[' + str(i) + ']') + "\n"
        prior += "\n" + "\t" + 'return log(prior);' + "\n" + '}'
        return prior

    def get_proposal_parameter_values(self):
        return_list = []
        for m, p in self._get_estimable_parameters_list():
            for param in p.sampling_proposal.get_parameters():
                if param.adaptable:
                    return_list.append(param.default_value)
        return return_list

    def is_proposal_symmetric(self):
        return all(p.sampling_proposal.is_symmetric for m, p in self._get_estimable_parameters_list())

    def get_proposal_logpdf(self, func_name='getProposalLogPDF'):
        return_str = ''
        for _, p in self._get_estimable_parameters_list():
            return_str += p.sampling_proposal.get_proposal_logpdf_function()

        return_str += "\n" + 'double ' + func_name + \
            '(const int i, const double proposal, const double current, double* const parameters){' + "\n\t"

        return_str += "\n\t" + 'switch(i){' + "\n\t\t"

        adaptable_parameter_count = 0
        for i, (m, p) in enumerate(self._get_estimable_parameters_list()):
            return_str += 'case ' + str(i) + ':' + "\n\t\t\t"

            param_proposal = p.sampling_proposal
            logpdf_call = 'return ' + param_proposal.get_proposal_logpdf_function_name() + '(proposal, current'

            for param in param_proposal.get_parameters():
                if param.adaptable:
                    logpdf_call += ', parameters[' + str(adaptable_parameter_count) + ']'
                    adaptable_parameter_count += 1
                else:
                    logpdf_call += ', ' + str(param.default_value)

            logpdf_call += ');'
            return_str += logpdf_call + "\n"

        return_str += "\n\t\t" + '}' + "\n" + 'return 0;' + "\n"
        return_str += '}'
        return return_str

    def get_proposal_function(self, func_name='getProposal'):
        return_str = ''
        for _, p in self._get_estimable_parameters_list():
            return_str += p.sampling_proposal.get_proposal_function()

        return_str += "\n" + 'double ' + func_name + \
            '(const int i, const double current, ranluxcl_state_t* const ranluxclstate, ' \
            ' double* const parameters){'

        return_str += "\n\t" + 'switch(i){' + "\n\t\t"

        adaptable_parameter_count = 0
        for i, (m, p) in enumerate(self._get_estimable_parameters_list()):
            return_str += 'case ' + str(i) + ':' + "\n\t\t\t"

            param_proposal = p.sampling_proposal
            proposal_call = 'return ' + param_proposal.get_proposal_function_name() + '(current, ranluxclstate'

            for param in param_proposal.get_parameters():
                if param.adaptable:
                    proposal_call += ', parameters[' + str(adaptable_parameter_count) + ']'
                    adaptable_parameter_count += 1
                else:
                    proposal_call += ', ' + str(param.default_value)

            proposal_call += ');'
            return_str += proposal_call + "\n"

        return_str += "\n\t\t" + '}' + "\n" + 'return 0;' + "\n"
        return_str += '}'
        return return_str

    def get_proposal_parameters_update_function(self, func_name='updateProposalParameters'):
        return_str = ''
        for _, p in self._get_estimable_parameters_list():
            for param in p.sampling_proposal.get_parameters():
                if param.adaptable:
                    return_str += param.get_parameter_update_function()

        return_str += 'void ' + func_name + '(uint* const ac_between_proposal_updates, ' + \
            'const uint proposal_update_intervals, double* const proposal_parameters){' + "\n"

        adaptable_parameter_count = 0
        for i, (m, p) in enumerate(self._get_estimable_parameters_list()):
            param_proposal = p.sampling_proposal

            for param in param_proposal.get_parameters():
                if param.adaptable:
                    return_str += "\t" * 3
                    return_str += 'proposal_parameters[' + str(adaptable_parameter_count) + '] = '
                    return_str += param.get_parameter_update_function_name() + '(' +\
                        'proposal_parameters[' + str(adaptable_parameter_count) + '], ' + \
                        'ac_between_proposal_updates[' + str(i) + '], proposal_update_intervals);' + "\n"
                    adaptable_parameter_count += 1

        return_str += '}'
        return return_str

    def get_log_likelihood_function(self, func_name="getLogLikelihood"):
        inst_per_problem = self.get_nmr_inst_per_problem()
        eval_func_name = func_name + '_evaluateModel'
        obs_func_name = func_name + '_getObservation'

        param_listing = ''
        for p in self._evaluation_model.get_free_parameters():
            param_listing += self._get_param_listing_for_param(self._evaluation_model, p)

        func = self.get_model_eval_function(eval_func_name)
        func += self.get_observation_return_function(obs_func_name)
        func += self._evaluation_model.get_log_likelihood_function(func_name, inst_per_problem, eval_func_name,
                                                                   obs_func_name, param_listing)
        return func

    def samples_to_statistics(self, samples_dict):
        results = {}
        for key, value in samples_dict.items():
            param = self._get_parameter_by_name(key)
            stat_mod = param.sampling_statistics
            results[key] = stat_mod.get_mean(value)
            results[key + '.std'] = stat_mod.get_std(value)
        return results


class DependencyStore(object):

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


class ParameterNameException(Exception):
    """Thrown when the a parameter of an given name could not be found."""
    pass


class ParameterResolutionException(Exception):
    """Thrown when a fixed parameter could not be resolved."""
    pass


class DoubleModelNameException(Exception):
    """Thrown when there are two models with the same name."""
    pass