from pkg_resources import resource_filename
from mot.cl_data_type import CLDataType
from mot.model_building.cl_functions.base import SimpleLibraryFunctionFromFile
from mot.model_building.cl_functions.parameters import LibraryParameter

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class FirstLegendreTerm(SimpleLibraryFunctionFromFile):

    def __init__(self):
        """A function for finding the first legendre term. (see the CL code for more details)
        """
        super(FirstLegendreTerm, self).__init__(
            'double',
            'getFirstLegendreTerm',
            (LibraryParameter(CLDataType.from_string('double'), 'x'),
             LibraryParameter(CLDataType.from_string('int'), 'n')),
            (),
            resource_filename('mot', 'data/opencl/firstLegendreTerm.h'),
            resource_filename('mot', 'data/opencl/firstLegendreTerm.cl'),
            {})


class Bessel(SimpleLibraryFunctionFromFile):

    def __init__(self):
        """Function library for the bessel functions.

        See the CL code for more details.
        """
        super(Bessel, self).__init__(
            'mot_float_type',
            '',
            (),
            (),
            resource_filename('mot', 'data/opencl/bessel.h'),
            resource_filename('mot', 'data/opencl/bessel.cl'),
            {})


class CerfImWOfX(SimpleLibraryFunctionFromFile):

    def __init__(self):
        """Calculate the cerf. (see the CL code for more details)
        """
        super(CerfImWOfX, self).__init__(
            'mot_float_type',
            'im_w_of_x',
            (LibraryParameter(CLDataType.from_string('mot_float_type'), 'x'),),
            (),
            resource_filename('mot', 'data/opencl/cerf/im_w_of_x.h'),
            resource_filename('mot', 'data/opencl/cerf/im_w_of_x.cl'),
            {})


class CerfDawson(SimpleLibraryFunctionFromFile):

    def __init__(self):
        """Evaluate dawson integral. (see the CL code for more details)
        """
        super(CerfDawson, self).__init__(
            'mot_float_type',
            'dawson',
            (LibraryParameter(CLDataType.from_string('mot_float_type'), 'x'),),
            (CerfImWOfX(),),
            resource_filename('mot', 'data/opencl/cerf/dawson.h'),
            resource_filename('mot', 'data/opencl/cerf/dawson.cl'),
            {})


class CerfErfi(SimpleLibraryFunctionFromFile):

    def __init__(self):
        """Calculate erfi. (see the CL code for more details)
        """
        super(CerfErfi, self).__init__(
            'mot_float_type',
            'erfi',
            (LibraryParameter(CLDataType.from_string('mot_float_type'), 'x'),),
            (CerfImWOfX(),),
            resource_filename('mot', 'data/opencl/cerf/erfi.h'),
            resource_filename('mot', 'data/opencl/cerf/erfi.cl'),
            {})


class EuclidianNormFunction(SimpleLibraryFunctionFromFile):

    def __init__(self, memspace='private', memtype='mot_float_type'):
        """A CL functions for calculating the Euclidian distance between n values.

        Args:
            memspace (str): The memory space of the memtyped array (private, constant, global).
            memtype (str): the memory type to use, double, float, mot_float_type, ...
        """
        super(EuclidianNormFunction, self).__init__(
            memtype,
            'euclidian_norm_' + memspace + '_' + memtype,
            (LibraryParameter(CLDataType.from_string(memtype + '*'), 'x'),
             LibraryParameter(CLDataType.from_string('int'), 'n')),
            (),
            resource_filename('mot', 'data/opencl/euclidian_norm.ph'),
            resource_filename('mot', 'data/opencl/euclidian_norm.pcl'),
            {'MEMSPACE': memspace, 'MEMTYPE': memtype})
