import numpy as np
import pyopencl as cl

from mot.cl_runtime_info import CLRuntimeInfo
from mot.kernel_data import Zeros
from mot.load_balance_strategies import Worker
from mot.utils import KernelDataManager, get_float_type_def

__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def apply_cl_function(cl_function, kernel_data, nmr_instances, use_local_reduction=False, cl_runtime_info=None):
    """Run the given function/procedure on the given set of data.

    This class will wrap the given CL function in a kernel call and execute that that for every data instance using
    the provided kernel data. This class will respect the read write setting of the kernel data elements such that
    output can be written back to the according kernel data elements.

    Args:
        cl_function (mot.cl_function.CLFunction): the function to
            run on the datasets. Either a name function tuple or an actual CLFunction object.
        kernel_data (dict[str: mot.utils.KernelData]): the data to use as input to the function
            all the data will be wrapped in a single ``mot_data_struct``.
        nmr_instances (int): the number of parallel threads to run (used as ``global_size``)
        use_local_reduction (boolean): set this to True if you want to use local memory reduction in
             your CL procedure. If this is set to True we will multiply the global size (given by the nmr_instances)
             by the work group sizes.
        cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the runtime information
    """
    cl_runtime_info = cl_runtime_info or CLRuntimeInfo()

    if cl_function.get_return_type() != 'void':
        kernel_data['_results'] = Zeros((nmr_instances,), cl_function.get_return_type())

    workers = []
    for cl_environment in cl_runtime_info.get_cl_environments():
        workers.append(_ProcedureWorker(cl_environment, cl_runtime_info.get_compile_flags(),
                                        cl_function,
                                        kernel_data, cl_runtime_info.double_precision, use_local_reduction))

    cl_runtime_info.load_balancer.process(workers, nmr_instances)

    if cl_function.get_return_type() != 'void':
        return kernel_data['_results'].get_data()


class _ProcedureWorker(Worker):

    def __init__(self, cl_environment, compile_flags, cl_function,
                 kernel_data, double_precision, use_local_reduction):
        super(_ProcedureWorker, self).__init__(cl_environment)
        self._cl_function = cl_function
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
                 (int(nmr_problems),),
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

        func_args = self._get_function_call_args()
        wrapped_arrays = self._get_wrapped_arrays()

        assignment = ''
        if self._cl_function.get_return_type() != 'void':
            assignment = '*(data._results) = '

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._data_struct_manager.get_struct_definition()
        kernel_source += self._cl_function.get_cl_code()
        kernel_source += '''
            __kernel void run_procedure(
                    ''' + ",\n".join(kernel_param_names) + '''){

                ulong gid = ''' + ('(ulong)(get_global_id(0) / get_local_size(0));'
                                   if self._use_local_reduction else 'get_global_id(0)') + ''';
                
                mot_data_struct data = ''' + self._data_struct_manager.get_struct_init_string('gid') + ''';
                
                ''' + wrapped_arrays + '''            
                ''' + assignment + ' ' + self._cl_function.get_cl_function_name() + '(' + ', '.join(func_args) + ');' \
            + '''
            }
        '''
        return kernel_source

    def _get_function_call_args(self):
        """Get the call arguments for calling the function we are wrapping in a kernel."""
        func_args = []
        for param in self._cl_function.get_parameters():
            param_cl_name = param.name.replace('.', '_')

            if param.data_type.raw_data_type == 'mot_data_struct':
                func_args.append('&data')
            elif self._kernel_data[param_cl_name].loaded_as_pointer:
                if param.data_type.is_pointer_type:
                    if param.data_type.address_space == 'private':
                        func_args.append(param_cl_name + '_private')
                    elif param.data_type.address_space == 'local':
                        func_args.append(param_cl_name + '_local')
                    else:
                        func_args.append('data.{}'.format(param_cl_name))
                else:
                    func_args.append('data.{}[0]'.format(param_cl_name))
            else:
                func_args.append('data.{}'.format(param_cl_name))

        return func_args

    def _get_wrapped_arrays(self):
        """For functions that require private arrays as input, change the address space of the global arrays.

        This creates a new array in the global address space and fills it with the values of the global array.

        This can be removed at the moment OpenCL 2.0 is supported (by using the generic address space).

        Returns:
            str: converts the address space of the input array from global to private, for those parameters that
                require it.
        """
        conversions = ''
        for parameter in self._cl_function.get_parameters():
            if parameter.data_type.raw_data_type == 'mot_data_struct':
                pass
            elif parameter.data_type.is_pointer_type:
                if parameter.data_type.address_space == 'private':
                    conversions += '''
                        {ctype} {param_name}_private[{nmr_elements}];
    
                        for(uint i = 0; i < {nmr_elements}; i++){{
                            {param_name}_private[i] = data.{param_name}[i];
                        }}
                    '''.format(ctype=parameter.data_type.ctype, param_name=parameter.name,
                               nmr_elements=self._kernel_data[parameter.name].data_length)
                elif parameter.data_type.address_space == 'local':
                    conversions += '''
                        local {ctype} {param_name}_local[{nmr_elements}];

                        for(uint i = 0; i < {nmr_elements}; i++){{
                            {param_name}_local[i] = data.{param_name}[i];
                        }}
                    '''.format(ctype=parameter.data_type.ctype, param_name=parameter.name,
                               nmr_elements=self._kernel_data[parameter.name].data_length)
        return conversions

