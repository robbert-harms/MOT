__author__ = 'Robbert Harms'
__date__ = '2017-06-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class ModelFunctionPrior(object):

    def get_prior_function(self):
        """Return the function that represents this prior.

        Returns:
            string: the cl for the prior of this function.
                This function must accept all the parameters listed by :meth:`get_function_parameters`.
                The signature is:

                .. code-block: c

                    mot_float_type <prior_fname>(mot_float_type p1, mot_float_type p2, ...);
        """
        raise NotImplementedError()

    def get_function_parameters(self):
        """Get a list of the parameters required in this prior (in that order).

        Returns:
            list of str: the list of parameter names this prior requires.
        """
        raise NotImplementedError()

    def get_prior_function_name(self):
        """Get the name of the prior function call.

         This is used by the model builder to construct the call to the prior function.

         Returns:
            str: name of the function
        """
        raise NotImplementedError()


class SimpleModelFunctionPrior(ModelFunctionPrior):

    def __init__(self, body, parameters, function_name):
        """Create a compartment prior from the function body and a list of its parameters.

        Args:
            body (str): the CL code of the function body
            parameters (list of str): the list with parameter names
            function_name (str): the name of this function
        """
        self._body = body
        self._parameters = parameters
        self._function_name = function_name

    def get_prior_function(self):
        return '''
            mot_float_type {function_name}({parameters}){{
                {body}
            }}
        '''.format(function_name=self._function_name,
                   parameters=', '.join('const mot_float_type {}'.format(p) for p in self._parameters),
                   body=self._body)

    def get_prior_function_name(self):
        return self._function_name

    def get_function_parameters(self):
        return self._parameters
