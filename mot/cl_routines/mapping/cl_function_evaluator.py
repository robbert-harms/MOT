import numpy as np

from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import SimpleNamedCLFunction, convert_data_to_dtype, SimpleKernelInputData
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

        Args:
            cl_function (mot.cl_function.CLFunction): the CL function to evaluate
            input_data (list of ndarray): a list with for each parameter of the model an input parameter
                to the function. Each of these input arrays must be of equal length in the first dimension.
            double_precision (boolean): if the function should be evaluated in double precision or not

        Returns:
            ndarray: a single array of the return type specified by the CL function,
                with for each parameter tuple an evaluation result
        """
        if not isinstance(input_data, (tuple, list)):
            input_data = [input_data]

        func_args = []
        for param in cl_function.get_parameters():
            if param.data_type.is_pointer_type:
                func_args.append('data->{}'.format(param.name))
            else:
                func_args.append('data->{}[0]'.format(param.name))

        func_name = 'evaluate'
        func = cl_function.get_cl_code()
        func += '''
            void ''' + func_name + '''(mot_data_struct* data){
                data->_results[0] = ''' + cl_function.get_cl_function_name() + '''(''' + ', '.join(func_args) + ''');  
            }
        '''
        named_func = SimpleNamedCLFunction(func, func_name)

        kernel_data = []
        for ind, param in enumerate(cl_function.get_parameters()):
            data = convert_data_to_dtype(input_data[ind], param.data_type.ctype,
                                         mot_float_type='double' if double_precision else 'float')
            kernel_data.append(SimpleKernelInputData(param.name, data))

        kernel_data.append(SimpleKernelInputData(
            '_results',
            convert_data_to_dtype(np.ones(input_data[0].shape[0]), cl_function.get_return_type(),
                                  mot_float_type='double' if double_precision else 'float'),
            is_writable=True))

        runner = RunProcedure(cl_environments=self.cl_environments, load_balancer=self.load_balancer,
                              compile_flags=self.compile_flags)
        runner.run_procedure(named_func, kernel_data, input_data[0].shape[0], double_precision=double_precision)

        return kernel_data[-1].get_data()
