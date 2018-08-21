import os
from textwrap import indent, dedent
from mot.lib.cl_function import CLFunction, SimpleCLFunction


__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLLibrary(CLFunction):
    pass


class SimpleCLLibrary(CLLibrary, SimpleCLFunction):

    def __init__(self, cl_code, **kwargs):
        func = SimpleCLFunction.from_string(cl_code, **kwargs)
        super().__init__(
            func.get_return_type(),
            func.get_cl_function_name(),
            func.get_parameters(),
            func.get_cl_body(),
            dependencies=func.get_dependencies(),
            cl_extra=func.get_cl_extra()
        )


class SimpleCLLibraryFromFile(CLLibrary, SimpleCLFunction):

    def __init__(self, return_type, cl_function_name, parameter_list, cl_code_file,
                 var_replace_dict=None, **kwargs):
        """Create a CL function for a library function.

        These functions are not meant to be optimized, but can be used a helper functions in models.

        Args:
            cl_function_name (str): The name of the CL function
            cl_code_file (str): The location of the code file
            var_replace_dict (dict): In the cl_code file these replacements will be made
                (using the % format function of Python)
        """
        with open(os.path.abspath(cl_code_file), 'r') as f:
            code = f.read()

        if var_replace_dict is not None:
            code = code % var_replace_dict

        super().__init__(return_type, cl_function_name, parameter_list, code, **kwargs)
        self._code = code

    def get_cl_code(self):
        return dedent('''
            {dependencies}
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            {cl_extra}
            {code}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=indent(self._get_cl_dependency_code(), ' ' * 4 * 3),
                   inclusion_guard_name='INCLUDE_GUARD_{}'.format(self.get_cl_function_name()),
                   cl_extra=self._cl_extra if self._cl_extra is not None else '',
                   code=indent('\n' + self._code.strip() + '\n', ' ' * 4 * 3)))
