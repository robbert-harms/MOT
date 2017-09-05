import pyopencl as cl
from ...utils import get_float_type_def, DataStructManager
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = '2017-08-31'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class RunProcedure(CLRoutine):

    def __init__(self, **kwargs):
        """This class can run any arbitrary given CL procedure on the given set of data.
        """
        super(RunProcedure, self).__init__(**kwargs)

    def run_procedure(self, named_cl_function, kernel_data, nmr_instances, double_precision=False):
        """Run the given function/procedure on the given set of data.

        This class will wrap the given CL function in a kernel call and execute that that for every data instance using
        the provided kernel data. This class will respect the read write setting of the kernel data elements such that
        output can be written back to the according kernel data elements.

        Args:
            named_cl_function (mot.utils.NamedCLFunction): the function to run on the datasets
            kernel_data (dict[str: mot.utils.KernelInputData]): the data to use as input to the function
                all the data will be wrapped in a single ``mot_data_struct``.
            nmr_instances (int): the number of parallel threads to run
            double_precision (boolean): if we want to run in double precision. Defaults to True.
        """
        workers = self._create_workers(lambda cl_environment: _ProcedureWorker(
            cl_environment, self.get_compile_flags_list(True),
            named_cl_function, kernel_data, double_precision))
        self.load_balancer.process(workers, nmr_instances)


class _ProcedureWorker(Worker):

    def __init__(self, cl_environment, compile_flags, named_cl_function, kernel_data, double_precision):
        super(_ProcedureWorker, self).__init__(cl_environment)
        self._cl_func = named_cl_function.get_cl_code()
        self._cl_func_name = named_cl_function.get_cl_function_name()
        self._kernel_data = kernel_data
        self._double_precision = double_precision

        self._data_struct_manager = DataStructManager(self._kernel_data)
        self._buffers = self._get_buffers()
        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def _get_buffers(self):
        buffers = []
        for data in [self._kernel_data[key] for key in sorted(self._kernel_data)]:
            if data.is_writable():
                buffers.append(cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                         hostbuf=data.get_data()))
            else:
                buffers.append(cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                         hostbuf=data.get_data()))
        return buffers

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        self._kernel.run_procedure(self._cl_run_context.queue, (int(nmr_problems), ), None,
                                   *self._buffers, global_offset=(int(range_start),))

        for ind, name in enumerate(sorted(self._kernel_data)):
            if self._kernel_data[name].is_writable():
                self._enqueue_readout(self._buffers[ind], self._kernel_data[name].get_data(), range_start, range_end)

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
