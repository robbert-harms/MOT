import numpy as np

from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import SimpleNamedCLFunction, convert_data_to_dtype, SimpleKernelInputData, KernelInputData
from ...cl_routines.base import CLRoutine


__author__ = 'Robbert Harms'
__date__ = '2017-08-31'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class CLFunctionEvaluator(CLRoutine):

    def __init__(self, **kwargs):
        """This class can evaluate an arbitrary CL function implementation on some input data.
        """
        super(CLFunctionEvaluator, self).__init__(**kwargs)

    def evaluate(self, cl_function, input_data, double_precision=False):
        """Evaluate the given CL function at the given data points.

        This function will convert possible dots in the parameter name to underscores for in the CL kernel.

        Args:
            cl_function (mot.cl_function.CLFunction): the CL function to evaluate
            input_data (dict): for each parameter of the function either an array with input data or an
                :class:`mot.utils.KernelInputData` object. Each of these input arrays must be of equal length
                in the first dimension.
            double_precision (boolean): if the function should be evaluated in double precision or not

        Returns:
            ndarray: a single array of the return type specified by the CL function,
                with for each parameter tuple an evaluation result
        """
        nmr_data_points = input_data[list(input_data)[0]].shape[0]

        kernel_items = self._wrap_kernel_items(cl_function, input_data, double_precision)
        kernel_items['_results'] = SimpleKernelInputData(
            convert_data_to_dtype(np.ones(nmr_data_points), cl_function.get_return_type(),
                                  mot_float_type='double' if double_precision else 'float'),
            is_writable=True)

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._wrap_cl_function(cl_function),
                             kernel_items, nmr_data_points, double_precision=double_precision)

        return kernel_items['_results'].get_data()

    def _wrap_kernel_items(self, cl_function, input_data, double_precision):
        kernel_items = {}
        for param in cl_function.get_parameters():
            if isinstance(input_data[param.name], KernelInputData):
                kernel_items[self._get_param_cl_name(param.name)] = input_data[param.name]
            else:
                data = convert_data_to_dtype(input_data[param.name], param.data_type.ctype,
                                             mot_float_type='double' if double_precision else 'float')
                kernel_items[self._get_param_cl_name(param.name)] = SimpleKernelInputData(data)
        return kernel_items

    def _wrap_cl_function(self, cl_function):
        func_args = []
        for param in cl_function.get_parameters():
            param_cl_name = self._get_param_cl_name(param.name)

            if param.data_type.is_pointer_type:
                func_args.append('data->{}'.format(param_cl_name))
            else:
                func_args.append('data->{}[0]'.format(param_cl_name))

        func_name = 'evaluate'
        func = cl_function.get_cl_code()
        func += '''
            void ''' + func_name + '''(mot_data_struct* data){
                
                printf("%f \\n", data->value[0]);
            
                data->_results[0] = ''' + cl_function.get_cl_function_name() + '''(''' + ', '.join(func_args) + ''');  
            }
        '''
        return SimpleNamedCLFunction(func, func_name)

    def _get_param_cl_name(self, param_name):
        if '.' in param_name:
            return param_name.replace('.', '_')
        return param_name
