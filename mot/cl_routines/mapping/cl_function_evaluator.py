import numpy as np

from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import NameFunctionTuple, is_scalar
from mot.kernel_data import KernelData, KernelScalar, KernelArray, KernelAllocatedArray
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
        nmr_data_points = self._get_minimum_data_length(input_data)

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
        min_data_length = self._get_minimum_data_length(input_data)

        kernel_items = {}
        for param in cl_function.get_parameters():
            if isinstance(input_data[param.name], KernelData):
                kernel_items[self._get_param_cl_name(param.name)] = input_data[param.name]
            elif is_scalar(input_data[param.name]) and not param.data_type.is_pointer_type:
                kernel_items[self._get_param_cl_name(param.name)] = KernelScalar(input_data[param.name])
            else:
                if is_scalar(input_data[param.name]):
                    data = np.ones(min_data_length) * input_data[param.name]
                else:
                    data = input_data[param.name]

                kernel_items[self._get_param_cl_name(param.name)] = KernelArray(
                    data, ctype=param.data_type.ctype, is_writable=True, is_readable=True)

        return kernel_items

    def _get_minimum_data_length(self, input_data):
        min_length = 1

        for value in input_data.values():
            if isinstance(value, KernelData):
                data = value.get_data()
                if data is not None:
                    if np.ndarray(data).shape[0] > min_length:
                        min_length = np.ndarray(data).shape[0]
            elif is_scalar(value):
                pass
            elif isinstance(value, (tuple, list)):
                min_length = len(value)
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
                    func_args.append(param_cl_name)
                else:
                    func_args.append('data->{}[0]'.format(param_cl_name))

        func_name = 'evaluate'
        func = cl_function.get_cl_code()

        wrapped_arrays = self._wrap_arrays(cl_function, kernel_items)

        if cl_function.get_return_type() == 'void':
            func += '''
                void ''' + func_name + '''(mot_data_struct* data){
                ''' + wrapped_arrays + '''
                    ''' + cl_function.get_cl_function_name() + '''(''' + ', '.join(func_args) + ''');  
                }
            '''
        else:
            func += '''
                void ''' + func_name + '''(mot_data_struct* data){
                    ''' + wrapped_arrays + '''
                    *(data->_results) = ''' + cl_function.get_cl_function_name() + '(' + ', '.join(func_args) + ''');  
                }
            '''
        return NameFunctionTuple(func_name, func)

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

        print(conversions)
        return conversions


    def _get_param_cl_name(self, param_name):
        if '.' in param_name:
            return param_name.replace('.', '_')
        return param_name
