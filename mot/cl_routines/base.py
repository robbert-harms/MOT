import numpy as np
import pyopencl as cl

from mot.cl_runtime_info import CLRuntimeInfo
from mot.kernel_data import KernelAllocatedArray, KernelData, KernelScalar, KernelArray
from mot.load_balance_strategies import Worker
from mot.utils import is_scalar, KernelDataManager, get_float_type_def

__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLRoutine(object):

    def __init__(self, cl_runtime_info=None):
        """Base class for CL routines.

        Args:
            cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the runtime information
        """
        self._cl_runtime_info = cl_runtime_info or CLRuntimeInfo()

    def set_cl_runtime_info(self, cl_runtime_info):
        """Update the CL runtime information.

        Args:
            cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the new runtime information
        """
        self._cl_runtime_info = cl_runtime_info


class CLFunctionEvaluator(CLRoutine):

    def __init__(self, **kwargs):
        """This class can evaluate an arbitrary CL function implementation on some input data.
        """
        super(CLFunctionEvaluator, self).__init__(**kwargs)

    def evaluate(self, cl_function, input_data, return_inputs=False):
        """Evaluate the given CL function at the given data points.

        This function will convert possible dots in the parameter name to underscores for in the CL kernel.

        Args:
            cl_function (mot.cl_function.CLFunction): the CL function to evaluate
            input_data (dict): for each parameter of the function either an array with input data or an
                :class:`mot.utils.KernelData` object. Each of these input datasets must either be a scalar or be
                of equal length in the first dimension. The user can either input raw ndarrays or input
                KernelData objects. If an ndarray is given we will load it read/write by default.
            return_inputs (boolean): if we are interested in the values of the input arrays after evaluation.

        Returns:
            ndarray or tuple(ndarray, dict[str: ndarray]): we always return at least the return values of the function,
                which can be None if this function has a void return type. If ``return_inputs`` is set to True then
                we return a tuple with as first element the return value and as second element a dictionary mapping
                the output state of the parameters.
        """
        for param in cl_function.get_parameters():
            if param.name not in input_data:
                names = [param.name for param in cl_function.get_parameters()]
                missing_names = [name for name in names if name not in input_data]
                raise ValueError('Some parameters are missing an input value, '
                                 'required parameters are: {}, missing inputs are: {}'.format(names, missing_names))

        nmr_data_points = self._get_minimum_data_length(cl_function, input_data)

        kernel_items = self._wrap_input_data(cl_function, input_data)

        if cl_function.get_return_type() != 'void':
            kernel_items['_results'] = KernelAllocatedArray((nmr_data_points,), cl_function.get_return_type())

        runner = RunProcedure(self._cl_runtime_info)
        runner.run_procedure(self._wrap_cl_function(cl_function, kernel_items), kernel_items, nmr_data_points)

        if cl_function.get_return_type() != 'void':
            return_value = kernel_items['_results'].get_data()
            del kernel_items['_results']
        else:
            return_value = None

        if return_inputs:
            return return_value, {key: value.get_data() for key, value in kernel_items.items()}
        return return_value

    def _wrap_input_data(self, cl_function, input_data):
        min_data_length = self._get_minimum_data_length(cl_function, input_data)

        def get_kernel_data(param):
            if isinstance(input_data[param.name], KernelData):
                return input_data[param.name]
            elif param.data_type.is_vector_type and np.squeeze(input_data[param.name]).shape[0] == 3:
                return KernelScalar(input_data[param.name], ctype=param.data_type.ctype)
            elif is_scalar(input_data[param.name]) and not param.data_type.is_pointer_type:
                return KernelScalar(input_data[param.name])
            else:
                if is_scalar(input_data[param.name]):
                    data = np.ones(min_data_length) * input_data[param.name]
                else:
                    data = input_data[param.name]

                return KernelArray(data, ctype=param.data_type.ctype, is_writable=True, is_readable=True)

        kernel_items = {}
        for param in cl_function.get_parameters():
            kernel_items[self._get_param_cl_name(param.name)] = get_kernel_data(param)

        return kernel_items

    def _get_minimum_data_length(self, cl_function, input_data):
        min_length = 1

        for param in cl_function.get_parameters():
            value = input_data[param.name]

            if isinstance(input_data[param.name], KernelData):
                data = value.get_data()
                if data is not None:
                    if np.ndarray(data).shape[0] > min_length:
                        min_length = np.maximum(min_length, np.ndarray(data).shape[0])

            elif param.data_type.is_vector_type and np.squeeze(input_data[param.name]).shape[0] == 3:
                pass
            elif is_scalar(input_data[param.name]):
                pass
            else:
                if isinstance(value, (tuple, list)):
                    min_length = np.maximum(min_length, len(value))
                elif value.shape[0] > min_length:
                    min_length = np.maximum(min_length, value.shape[0])

        return min_length

    def _wrap_cl_function(self, cl_function, kernel_items):
        func_args = []
        for param in cl_function.get_parameters():
            param_cl_name = self._get_param_cl_name(param.name)

            if kernel_items[param_cl_name].is_scalar:
                func_args.append('data->{}'.format(param_cl_name))
            else:
                if param.data_type.is_pointer_type:
                    func_args.append(param_cl_name)
                else:
                    func_args.append('data->{}[0]'.format(param_cl_name))

        func_name = 'evaluate'
        wrapped_arrays = self._wrap_arrays(cl_function, kernel_items)

        if cl_function.get_return_type() == 'void':
            cl_body = '''
                ''' + wrapped_arrays + '''
                ''' + cl_function.get_cl_function_name() + '''(''' + ', '.join(func_args) + ''');  
            '''
        else:
            cl_body = '''
                ''' + wrapped_arrays + '''
                *(data->_results) = ''' + cl_function.get_cl_function_name() + '(' + ', '.join(func_args) + ''');  
            '''

        from mot.cl_function import SimpleCLFunction
        return SimpleCLFunction('void', func_name, ['mot_data_struct* data'], cl_body, dependencies=[cl_function])

    def _wrap_arrays(self, cl_function, kernel_items):
        """For functions that require private arrays as input, change the address space of the global arrays.

        This does not actually change the address space, but creates a new array in the global address space and
        fills it with the values of the global array.

        Returns:
            str: converts the address space of the input array from global to private, for those parameters that
                require it.
        """
        conversions = ''

        parameters = cl_function.get_parameters()
        for parameter in parameters:
            if parameter.data_type.is_pointer_type:
                if parameter.data_type.address_space == 'private':
                    conversions += '''
                        {ctype} {param_name}[{nmr_elements}];
                        
                        for(uint i = 0; i < {nmr_elements}; i++){{
                            {param_name}[i] = data->{param_name}[i];
                        }}
                    '''.format(ctype=parameter.data_type.ctype, param_name=parameter.name,
                               nmr_elements=kernel_items[parameter.name].data_length)
        return conversions

    def _get_param_cl_name(self, param_name):
        if '.' in param_name:
            return param_name.replace('.', '_')
        return param_name


# def apply_cl_function(cl_function, kernel_data, nmr_instances, use_local_reduction=False, cl_runtime_info=None):
#     """Run the given function/procedure on the given set of data.
#
#     This class will wrap the given CL function in a kernel call and execute that that for every data instance using
#     the provided kernel data. This class will respect the read write setting of the kernel data elements such that
#     output can be written back to the according kernel data elements.
#
#     Args:
#         cl_function (mot.cl_function.CLFunction): the function to
#             run on the datasets. Either a name function tuple or an actual CLFunction object.
#         kernel_data (dict[str: mot.utils.KernelData]): the data to use as input to the function
#             all the data will be wrapped in a single ``mot_data_struct``.
#         nmr_instances (int): the number of parallel threads to run (used as ``global_size``)
#         use_local_reduction (boolean): set this to True if you want to use local memory reduction in
#              your CL procedure. If this is set to True we will multiply the global size (given by the nmr_instances)
#              by the work group sizes.
#         cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the runtime information
#     """
#     class _ProcedureWorker(Worker):
#         def __init__(self, cl_environment, compile_flags, named_cl_function,
#                      kernel_data, double_precision,use_local_reduction):
#             super(_ProcedureWorker, self).__init__(cl_environment)
#             self._cl_func = named_cl_function.get_cl_code()
#             self._cl_func_name = named_cl_function.get_cl_function_name()
#             self._kernel_data = kernel_data
#             self._double_precision = double_precision
#             self._use_local_reduction = use_local_reduction
#
#             mot_float_dtype = np.float32
#             if double_precision:
#                 mot_float_dtype = np.float64
#
#             self._data_struct_manager = KernelDataManager(self._kernel_data, mot_float_dtype)
#             self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)
#             self._workgroup_size = self._kernel.run_procedure.get_work_group_info(
#                 cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
#                 self._cl_environment.device)
#             if not self._use_local_reduction:
#                 self._workgroup_size = None
#
#             self._kernel_input = self._get_kernel_input()
#
#         def _get_kernel_input(self):
#             return self._data_struct_manager.get_kernel_inputs(self._cl_context, self._workgroup_size)
#
#         def calculate(self, range_start, range_end):
#             nmr_problems = range_end - range_start
#
#             func = self._kernel.run_procedure
#             func.set_scalar_arg_dtypes(self._data_struct_manager.get_scalar_arg_dtypes())
#
#             if self._workgroup_size is None:
#                 func(self._cl_queue,
#                      (int(nmr_problems),),
#                      None,
#                      *self._kernel_input,
#                      global_offset=(int(range_start),))
#             else:
#                 func(self._cl_queue,
#                      (int(nmr_problems * self._workgroup_size),),
#                      (int(self._workgroup_size),),
#                      *self._kernel_input,
#                      global_offset=(int(range_start * self._workgroup_size),))
#
#             for ind, name in self._data_struct_manager.get_items_to_write_out():
#                 self._enqueue_readout(self._kernel_input[ind], self._kernel_data[name].get_data(),
#                                       range_start, range_end)
#
#         def _get_kernel_source(self):
#             kernel_param_names = self._data_struct_manager.get_kernel_arguments()
#
#             kernel_source = ''
#             kernel_source += get_float_type_def(self._double_precision)
#             kernel_source += self._data_struct_manager.get_struct_definition()
#             kernel_source += self._cl_func
#             kernel_source += '''
#                 __kernel void run_procedure(
#                         ''' + ",\n".join(kernel_param_names) + '''){
#
#                     ulong gid = ''' + ('(ulong)(get_global_id(0) / get_local_size(0));'
#                                        if self._use_local_reduction else 'get_global_id(0)') + ''';
#
#                     mot_data_struct data = ''' + self._data_struct_manager.get_struct_init_string('gid') + ''';
#                     ''' + self._cl_func_name + '''(&data);
#                 }
#             '''
#             return kernel_source
#
#     cl_runtime_info = cl_runtime_info or CLRuntimeInfo()
#
#     workers = []
#     for cl_environment in cl_runtime_info.get_cl_environments():
#         workers.append(_ProcedureWorker(cl_environment, cl_runtime_info.get_compile_flags(), cl_function,
#                                         kernel_data, cl_runtime_info.double_precision, use_local_reduction))
#
#     cl_runtime_info.load_balancer.process(workers, nmr_instances)


class RunProcedure(CLRoutine):

    def __init__(self, *args, **kwargs):
        """This class can run any arbitrary given CL procedure on the given set of data."""
        super(RunProcedure, self).__init__(*args, **kwargs)

    def run_procedure(self, cl_function, kernel_data, nmr_instances, use_local_reduction=False):
        """Run the given function/procedure on the given set of data.

        This class will wrap the given CL function in a kernel call and execute that that for every data instance using
        the provided kernel data. This class will respect the read write setting of the kernel data elements such that
        output can be written back to the according kernel data elements.

        Args:
            cl_function (mot.cl_function.CLFunction): the function to run on the datasets
            kernel_data (dict[str: mot.utils.KernelData]): the data to use as input to the function
                all the data will be wrapped in a single ``mot_data_struct``.
            nmr_instances (int): the number of parallel threads to run (used as ``global_size``)
            use_local_reduction (boolean): set this to True if you want to use local memory reduction in
                 your CL procedure. If this is set to True we will multiply the global size (given by the nmr_instances)
                 by the work group sizes.
        """
        cl_environments = self._cl_runtime_info.get_cl_environments()
        workers = []
        for cl_environment in cl_environments:
            workers.append(self._ProcedureWorker(
                cl_environment, self._cl_runtime_info.get_compile_flags(),
                cl_function, kernel_data, self._cl_runtime_info.double_precision, use_local_reduction))

        self._cl_runtime_info.load_balancer.process(workers, nmr_instances)

    class _ProcedureWorker(Worker):

        def __init__(self, cl_environment, compile_flags, named_cl_function, kernel_data, double_precision,
                     use_local_reduction):
            super(RunProcedure._ProcedureWorker, self).__init__(cl_environment)
            self._cl_func = named_cl_function.get_cl_code()
            self._cl_func_name = named_cl_function.get_cl_function_name()
            self._kernel_data = kernel_data
            self._double_precision = double_precision
            self._use_local_reduction = use_local_reduction

            mot_float_dtype = np.float32
            if double_precision:
                mot_float_dtype = np.float64

            self._data_struct_manager = KernelDataManager(self._kernel_data, mot_float_dtype)
            self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)
            self._workgroup_size = self._kernel.run_procedure.get_work_group_info(
                cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                self._cl_environment.device)
            if not self._use_local_reduction:
                self._workgroup_size = None

            self._kernel_input = self._get_kernel_input()

        def _get_kernel_input(self):
            return self._data_struct_manager.get_kernel_inputs(self._cl_context, self._workgroup_size)

        def calculate(self, range_start, range_end):
            nmr_problems = range_end - range_start

            func = self._kernel.run_procedure
            func.set_scalar_arg_dtypes(self._data_struct_manager.get_scalar_arg_dtypes())

            if self._workgroup_size is None:
                func(self._cl_queue,
                     (int(nmr_problems), ),
                     None,
                     *self._kernel_input,
                     global_offset=(int(range_start),))
            else:
                func(self._cl_queue,
                     (int(nmr_problems * self._workgroup_size),),
                     (int(self._workgroup_size),),
                     *self._kernel_input,
                     global_offset=(int(range_start * self._workgroup_size),))

            for ind, name in self._data_struct_manager.get_items_to_write_out():
                self._enqueue_readout(self._kernel_input[ind], self._kernel_data[name].get_data(),
                                      range_start, range_end)

        def _get_kernel_source(self):
            kernel_param_names = self._data_struct_manager.get_kernel_arguments()

            kernel_source = ''
            kernel_source += get_float_type_def(self._double_precision)
            kernel_source += self._data_struct_manager.get_struct_definition()
            kernel_source += self._cl_func
            kernel_source += '''
                __kernel void run_procedure(
                        ''' + ",\n".join(kernel_param_names) + '''){
    
                    ulong gid = ''' + ('(ulong)(get_global_id(0) / get_local_size(0));'
                                       if self._use_local_reduction else 'get_global_id(0)') + ''';
    
                    mot_data_struct data = ''' + self._data_struct_manager.get_struct_init_string('gid') + ''';
                    ''' + self._cl_func_name + '''(&data);
                }
            '''
            return kernel_source
