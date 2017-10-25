import numpy as np

from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import SimpleNamedCLFunction, convert_data_to_dtype, KernelInputArray, KernelInputData, is_scalar, \
    KernelInputScalar, KernelInputAllocatedOutput
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

    def evaluate(self, cl_function, input_data, double_precision=False, return_inputs=False):
        """Evaluate the given CL function at the given data points.

        This function will convert possible dots in the parameter name to underscores for in the CL kernel.

        Args:
            cl_function (mot.cl_function.CLFunction): the CL function to evaluate
            input_data (dict): for each parameter of the function either an array with input data or an
                :class:`mot.utils.KernelInputData` object. Each of these input datasets must either be a scalar or be
                of equal length in the first dimension. The user can either input raw ndarrays or input
                KernelInputData objects. If an ndarray is given we will load it read/write by default.
            double_precision (boolean): if the function should be evaluated in double precision or not
            return_inputs (boolean): if we are interested in the values of the input arrays after evaluation.

        Returns:
            ndarray or tuple(ndarray, dict[str: ndarray]): we always return at least the return values of the function,
                which can be None if this function has a void return type. If ``return_inputs`` is set to True then
                we return a tuple with as first element the return value and as second element a dictionary mapping
                the output state of the parameters.
        """
        nmr_data_points = self._get_minimum_data_length(input_data)

        kernel_items = self._wrap_input_data(cl_function, input_data, double_precision)

        if cl_function.get_return_type() != 'void':
            kernel_items['_results'] = KernelInputAllocatedOutput((nmr_data_points,), cl_function.get_return_type(),
                                                                  is_readable=False)

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._wrap_cl_function(cl_function, kernel_items),
                             kernel_items, nmr_data_points, double_precision=double_precision)

        if cl_function.get_return_type() != 'void':
            return_value = kernel_items['_results'].get_data()
            del kernel_items['_results']
        else:
            return_value = None

        if return_inputs:
            return return_value, {key: value.get_data() for key, value in kernel_items.items()}
        return return_value

    def _wrap_input_data(self, cl_function, input_data, double_precision):
        min_data_length = self._get_minimum_data_length(input_data)

        kernel_items = {}
        for param in cl_function.get_parameters():
            if isinstance(input_data[param.name], KernelInputData):
                kernel_items[self._get_param_cl_name(param.name)] = input_data[param.name]
            elif is_scalar(input_data[param.name]) and not param.data_type.is_pointer_type:
                kernel_items[self._get_param_cl_name(param.name)] = KernelInputScalar(input_data[param.name])
            else:
                if is_scalar(input_data[param.name]):
                    data = np.ones(min_data_length) * input_data[param.name]
                else:
                    data = input_data[param.name]

                kernel_items[self._get_param_cl_name(param.name)] = KernelInputArray(
                    data, ctype=param.data_type.ctype, is_writable=True, is_readable=True)

        return kernel_items

    def _get_minimum_data_length(self, input_data):
        min_length = 1

        for value in input_data.values():
            if isinstance(value, KernelInputData):
                data = value.get_data()
                if data is not None:
                    if np.ndarray(data).shape[0] > min_length:
                        min_length = np.ndarray(data).shape[0]
            elif is_scalar(value):
                pass
            elif value.shape[0] > min_length:
                min_length = value.shape[0]

        return min_length

    def _wrap_cl_function(self, cl_function, kernel_items):
        func_args = []
        for param in cl_function.get_parameters():
            param_cl_name = self._get_param_cl_name(param.name)

            if kernel_items[param_cl_name].is_scalar:
                func_args.append('data->{}'.format(param_cl_name))
            else:
                if param.data_type.is_pointer_type:
                    func_args.append('data->{}'.format(param_cl_name))
                else:
                    func_args.append('data->{}[0]'.format(param_cl_name))

        func_name = 'evaluate'
        func = cl_function.get_cl_code()

        if cl_function.get_return_type() == 'void':
            func += '''
                void ''' + func_name + '''(mot_data_struct* data){
                    ''' + cl_function.get_cl_function_name() + '''(''' + ', '.join(func_args) + ''');  
                }
            '''
        else:
            func += '''
                void ''' + func_name + '''(mot_data_struct* data){
                    *(data->_results) = ''' + cl_function.get_cl_function_name() + '(' + ', '.join(func_args) + ''');  
                }
            '''
        return SimpleNamedCLFunction(func, func_name)

    def _get_param_cl_name(self, param_name):
        if '.' in param_name:
            return param_name.replace('.', '_')
        return param_name
