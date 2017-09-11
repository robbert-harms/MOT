from ...utils import get_float_type_def, KernelInputDataManager
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = '2017-08-31'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class RunProcedure(CLRoutine):

    def __init__(self, **kwargs):
        """This class can run any arbitrary given CL procedure on the given set of data."""
        super(RunProcedure, self).__init__(**kwargs)

    def run_procedure(self, cl_function, kernel_data, nmr_instances, double_precision=False):
        """Run the given function/procedure on the given set of data.

        This class will wrap the given CL function in a kernel call and execute that that for every data instance using
        the provided kernel data. This class will respect the read write setting of the kernel data elements such that
        output can be written back to the according kernel data elements.

        Args:
            cl_function (mot.utils.NamedCLFunction): the function to run on the datasets
            kernel_data (dict[str: mot.utils.KernelInputData]): the data to use as input to the function
                all the data will be wrapped in a single ``mot_data_struct``.
            nmr_instances (int): the number of parallel threads to run
            double_precision (boolean): if we want to run in double precision. Defaults to True.
        """
        workers = self._create_workers(lambda cl_environment: _ProcedureWorker(
            cl_environment, self.get_compile_flags_list(True),
            cl_function, kernel_data, double_precision))
        self.load_balancer.process(workers, nmr_instances)


class _ProcedureWorker(Worker):

    def __init__(self, cl_environment, compile_flags, named_cl_function, kernel_data, double_precision):
        super(_ProcedureWorker, self).__init__(cl_environment)
        self._cl_func = named_cl_function.get_cl_code()
        self._cl_func_name = named_cl_function.get_cl_function_name()
        self._kernel_data = kernel_data
        self._double_precision = double_precision

        self._data_struct_manager = KernelInputDataManager(self._kernel_data)
        self._kernel_input = self._get_kernel_input()
        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def _get_kernel_input(self):
        return self._data_struct_manager.get_kernel_inputs(self._cl_run_context.context)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        func = self._kernel.run_procedure
        func.set_scalar_arg_dtypes(self._data_struct_manager.get_scalar_arg_dtypes())
        func(self._cl_run_context.queue, (int(nmr_problems), ), None, *self._kernel_input,
             global_offset=(int(range_start),))

        for ind, name in self._data_struct_manager.get_items_to_write_out():
            self._enqueue_readout(self._kernel_input[ind], self._kernel_data[name].get_data(), range_start, range_end)

    def _get_kernel_source(self):
        kernel_param_names = self._data_struct_manager.get_kernel_arguments()

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._data_struct_manager.get_struct_definition()
        kernel_source += self._cl_func
        kernel_source += '''
            __kernel void run_procedure(
                    ''' + ",\n".join(kernel_param_names) + '''){
                
                ulong gid = get_global_id(0);
                mot_data_struct data = ''' + self._data_struct_manager.get_struct_init_string('gid') + ''';
                ''' + self._cl_func_name + '''(&data);
            }
        '''
        return kernel_source
