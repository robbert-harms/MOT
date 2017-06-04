__author__ = 'Robbert Harms'
__date__ = '2017-05-29'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class ParameterCodec(object):

    def get_parameter_encode_function(self, fname='encodeParameters'):
        """Get a CL function that can transform the model parameters from model space to an encoded space.

        The signature of the CL function is:

        .. code-block:: c

            void <fname>(const void* data, const mot_float_type* x);

        Args:
            fname (str): The CL function name to use

        Returns:
            str: An OpenCL function that is used in the CL kernel to transform the parameters from model space to
                encoded space so they can be used as input to an CL routine.
        """
        raise NotImplementedError()

    def get_parameter_decode_function(self, fname='decodeParameters'):
        """Get a CL function that can transform the model parameters from encoded space to model space.

        The signature of the CL function is:

        .. code-block:: c

            void <fname>(const void* data, const mot_float_type* x);

        Args:
            fname (str): The CL function name to use

        Returns:
            str: An OpenCL function that is used in the CL kernel to transform the parameters from encoded space to
                model space so they can be used as input to the model.
        """
        raise NotImplementedError()


class ModelPrior(object):
    """Priors for an entire model. This is used in addition to the parameter priors."""

    def get_prior_function(self):
        """Get the CL code for the prior.

        Returns:
            str: The prior, with signature:

                .. code-block: c

                    mot_float_type <prior_fname>(mot_float_type p1, mot_float_type p2, ...);
        """
        raise NotImplementedError()

    def get_function_parameters(self):
        """Get the list of function parameters this prior requires.

        Returns:
            list of str: the parameter names in dot format
        """
        raise NotImplementedError()

    def get_function_name(self):
        """Get the name of this prior function

        Returns:
            str: the name of this function
        """


class SimpleModelPrior(ModelPrior):

    def __init__(self, body, parameters, function_name):
        """Easy construct a model prior.

        Args:
            body (str): the function body
            parameters (list of str): the list of parameter names used in this function (in dot format)
            function_name (str): the name of this prior function
        """
        self._body = body
        self._parameters = parameters
        self._function_name = function_name

    def get_prior_function(self):
        return '''
            mot_float_type {fname}({parameters}){{
                {body}
            }}
        '''.format(fname=self._function_name,
                   parameters=', '.join(['const mot_float_type {}'.format(p.replace('.', '_'))
                                         for p in self._parameters]),
                   body=self._body)

    def get_function_parameters(self):
        return self._parameters

    def get_function_name(self):
        return self._function_name
