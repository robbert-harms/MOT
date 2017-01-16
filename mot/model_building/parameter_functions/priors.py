__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractParameterPrior(object):
    """The priors are used during model sampling.

    They indicate the a priori information one has about a parameter.
    """

    def get_cl_assignment(self):
        """Get the assignment constructor.

        Returns:
            AssignmentConstructor: The assignment construction method for this prior
        """
        return ''


class UniformPrior(AbstractParameterPrior):
    """The uniform prior is always 1."""
    def get_cl_assignment(self):
        return FormatAssignmentConstructor('1')


class UniformWithinBoundsPrior(AbstractParameterPrior):
    """This prior is 1 within the upper and lower bound of the parameter, 0 outside."""
    def get_cl_assignment(self):
        return FormatAssignmentConstructor('(({parameter_variable} < {lower_bound} '
                                           '  || {parameter_variable} > {upper_bound}) ? 0.0 : 1.0)')


class AbsSinPrior(AbstractParameterPrior):
    """The fabs(sin(x)) prior."""
    def get_cl_assignment(self):
        return FormatAssignmentConstructor('fabs(sin({parameter_variable}))')


class AbsSinHalfPrior(AbstractParameterPrior):
    """The fabs(sin(x)/2.0) prior. Taken from FSL"""
    def get_cl_assignment(self):
        return FormatAssignmentConstructor('fabs(sin({parameter_variable})/2.0)')


class AssignmentConstructor(object):

    def create_assignment(self, parameter_variable, lower_bound, upper_bound):
        """Create the assignment string.

        Args:
            parameter_variable (str): the name of the parameter variable holding the current value in the kernel
            lower_bound (str): the value or the name of the variable holding the value for the lower bound
            upper_bound (str): the value or the name of the variable holding the value for the upper bound

        Returns:
            str: the prior assignment
        """


class FormatAssignmentConstructor(AssignmentConstructor):

    def __init__(self, assignment):
        """Assignment constructor that formats the given assignment template.

        This expects that the assignment string has elements like:

        * ``{parameter_variable}``: for the parameter variable
        * ``{lower_bound}``: for the lower bound
        * ``{upper_bound}``: for the upper bound

        Args:
            assignment (str): the string containing the assignment template.
        """
        self._assignment = assignment

    def create_assignment(self, parameter_variable, lower_bound, upper_bound):
        assignment = self._assignment.replace('{parameter_variable}', parameter_variable)
        assignment = assignment.replace('{lower_bound}', lower_bound)
        assignment = assignment.replace('{upper_bound}', upper_bound)
        return assignment
