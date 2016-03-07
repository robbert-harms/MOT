import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2014-10-23"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ParameterSampleStatistics(object):
    def get_mean(self, samples):
        """Given the distribution represented by this statistic, get the mean of the samples.

        Args:
            samples (ndarray): The 2d array with the samples per voxel.

        Returns:
            A 1d ndarray with the mean per voxel.
        """

    def get_std(self, samples):
        """Given the distribution represented by this statistic, get the standard deviation of the samples.

        Args:
            samples (ndarray): The 2d array with the samples per voxel.

        Returns:
            A 1d array with the variance per voxel.
        """


class GaussianPSS(ParameterSampleStatistics):
    def get_mean(self, samples):
        return np.mean(samples, axis=1)

    def get_std(self, samples):
        return np.std(samples, axis=1)


class CircularGaussianPSS(ParameterSampleStatistics):
    def __init__(self, max_angle=np.pi):
        """Compute the circular mean for samples in a range

        The minimum angle is set to 0, the maximum angle can be given.

        Args:
            max_angle (number): The maximum angle used in the calculations
        """
        super(CircularGaussianPSS, self).__init__()
        self.max_angle = max_angle

    def get_mean(self, samples):
        return CircularGaussianPSS.circmean(np.mod(samples, self.max_angle), high=self.max_angle, low=0, axis=1)

    def get_std(self, samples):
        return CircularGaussianPSS.circstd(np.mod(samples, self.max_angle), high=self.max_angle, low=0, axis=1)

    @staticmethod
    def circmean(samples, high=2 * np.pi, low=0, axis=None):
        """Compute the circular mean for samples in a range.
        Taken from scipy.stats

        Args:
            samples (array_like): Input array.
            high (float or int): High boundary for circular mean range.  Default is ``2*pi``.
            low (float or int): Low boundary for circular mean range.  Default is 0.
            axis (int, optional): Axis along which means are computed.
                The default is to compute the mean of the flattened array.

        Returns:
            float: Circular mean.
        """
        ang = (samples - low) * 2 * np.pi / (high - low)
        res = np.angle(np.mean(np.exp(1j * ang), axis=axis))
        mask = res < 0
        if mask.ndim > 0:
            res[mask] += 2 * np.pi
        elif mask:
            res += 2 * np.pi
        return res * (high - low) / 2.0 / np.pi + low

    @staticmethod
    def circstd(samples, high=2 * np.pi, low=0, axis=None):
        """Compute the circular standard deviation for samples assumed to be in the range [low to high].

        Taken from scipy.stats, with a small change on the 4th line.

        This uses a definition of circular standard deviation that in the limit of
        small angles returns a number close to the 'linear' standard deviation.

        Args:
            samples (array_like): Input array.
            low (float or int): Low boundary for circular standard deviation range.  Default is 0.
            high (float or int): High boundary for circular standard deviation range. Default is ``2*pi``.
            axis (int): Axis along which standard deviations are computed.  The default is
                to compute the standard deviation of the flattened array.

        Returns:
            float: Circular standard deviation.
        """
        ang = (samples - low) * 2 * np.pi / (high - low)
        res = np.mean(np.exp(1j * ang), axis=axis)
        R = abs(res)
        R[R >= 1] = 1 - np.finfo(np.float).eps
        return ((high - low) / 2.0 / np.pi) * np.sqrt(-2 * np.log(R))
