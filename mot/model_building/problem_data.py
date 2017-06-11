import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractProblemData(object):
    """A simple data container for the data for optimization/sampling models."""

    @property
    def protocol(self):
        """Return the protocol data stored in this problem data container.

        The protocol data contains information about the experimental setup. In MRI this is the scanner protocol.

        Returns:
            collections.Mapping: The protocol data information mapping.
        """
        return {}

    def get_nmr_inst_per_problem(self):
        """Get the number of instances/data points per problem.

        The minimum is one instance per problem.

        This number represents the number of data points

        Returns:
            int: the number of instances per problem.
        """
        return np.array(self.protocol[list(self.protocol.keys())[0]]).shape[0]

    def get_nmr_problems(self):
        """Get the number of problems present in this problem data.

        Returns:
            int: the number of problem instances
        """
        return self.observations.shape[0]

    @property
    def observations(self):
        """Return the observations stored in this problem data container.

        Returns:
            ndarray: The list of observed instances per problem. Should be a 2d matrix with as columns the observations
                and as rows the problems.
        """
        return np.array([[]])

    @property
    def static_maps(self):
        """Get a dictionary with the static maps.

        These maps will be loaded by the model builder as the values for the static parameters.

        Returns:
            dict: per static map the value for the static map. This can either be an one or two dimensional
                matrix containing the values for each problem instance or it can be a single value we will use
                for all p
        """
        return {}

    @property
    def noise_std(self):
        """The noise standard deviation we will use during model evaluation.

        During optimization or sampling the model will be evaluated against the observations using an evaluation
        model. Most of these evaluation models need to have a standard deviation.

        Returns:
            number of ndarray: either a scalar or a 2d matrix with one value per problem instance.
        """
        return 1


class SimpleProblemData(AbstractProblemData):

    def __init__(self, protocol, observations, static_maps=None, noise_std=None):
        """A simple data container for the data for optimization/sampling models.

        Args:
            protocol (dict): The protocol data dictionary
            observations (ndarray): The 2d array with the observations
            static_maps (dict): The dictionary with the static maps. These are 2d/3d ndarrays with one or more
                values per problem instance.
            noise_std (number or ndarray): either a scalar or a 2d matrix with one value per problem instance.
        """
        self._protocol = protocol
        self._observation = observations
        self._static_maps = static_maps or {}
        self._noise_std = noise_std

    @property
    def protocol(self):
        return self._protocol

    @property
    def observations(self):
        return self._observation

    @property
    def static_maps(self):
        return self._static_maps

    @property
    def noise_std(self):
        return self._noise_std
