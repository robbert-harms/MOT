__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractParameterPrior(object):
    """The priors are used during model sampling.

    They indicate the a priori information one has about a parameter.
    """

    def get_cl_assignment(self, parameter, parameter_name):
        """Get the assignment code.

        In CL assignments look like: a = b;
        This function should return the "b;" part. That is, all that follows after the assignment operator.

        Args:
            parameter (FreeParameter): the free parameter for which to create a prior
            parameter_name (str): the name of the parameter for which we create the prior

        Returns:
            str: The assignment code for the prior
        """
        return ''


class UniformPrior(AbstractParameterPrior):
    """The uniform prior is always 1."""
    def get_cl_assignment(self, parameter, parameter_name):
        return '1;'


class UniformWithinBoundsPrior(AbstractParameterPrior):
    """This prior is 1 within the upper and lower bound of the parameter, 0 outside."""
    def get_cl_assignment(self, parameter, parameter_name):
        return '((' + parameter_name + ' < ' + str(float(parameter.lower_bound)) + \
               ' || ' + parameter_name + ' > ' + str(float(parameter.upper_bound)) + ') ? 0.0 : 1.0);'


class AbsSinPrior(AbstractParameterPrior):
    """The fabs(sin(x)) prior."""
    def get_cl_assignment(self, parameter, parameter_name):
        return 'fabs(sin(' + parameter_name + '));'


class AbsSinHalfPrior(AbstractParameterPrior):
    """The fabs(sin(x)/2.0) prior. Taken from FSL"""
    def get_cl_assignment(self, parameter, parameter_name):
        return 'fabs(sin(' + parameter_name + ')/2.0);'
