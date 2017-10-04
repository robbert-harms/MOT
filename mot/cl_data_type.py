import six

__author__ = 'Robbert Harms'
__date__ = "2015-03-21"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLDataType(object):
    """Interface for CL data type containers.

    Basically this encapsulates the type and its qualifiers that define a variable in CL.
    """

    def get_declaration(self):
        """Get the complete CL declaration for this datatype.

        Returns:
            str: the declaration for this data type.
        """
        raise NotImplementedError()

    @property
    def declaration_type(self):
        """Get the type of this parameter in CL language

        This only returns the parameter type (like ``double`` or ``int*`` or ``float4*`` ...). It does not include other
        qualifiers.

        Returns:
            str: The name of this data type
        """
        raise NotImplementedError()

    @property
    def raw_data_type(self):
        """Get the raw data type without the vector and pointer additions.

        For example, if the data type is float4*, we will only return float here.

        Returns:
            str: the raw CL data type
        """
        raise NotImplementedError()

    @property
    def ctype(self):
        """Get the ctype of this data type.

        Returns:
            str: the full ctype of this data type
        """
        raise NotImplementedError()

    @property
    def is_vector_type(self):
        """Check if this data type is a vector type (like for example double4, float2, int8, etc.).

        Returns:
            boolean: True if it is a vector type, false otherwise
        """
        raise NotImplementedError()

    @property
    def is_pointer_type(self):
        """Check if this parameter is a pointer type (appended by a ``*``)

        Returns:
            boolean: True if it is a pointer type, false otherwise
        """
        raise NotImplementedError()

    @property
    def vector_length(self):
        """Get the length of this vector, returns None if not a vector type.

        Returns:
            int: the length of the vector type (for example, if the data type is float4, this returns 4).
        """
        raise NotImplementedError()


class SimpleCLDataType(CLDataType):

    def __init__(self, raw_data_type, is_pointer_type=False, vector_length=None,
                 address_space_qualifier=None, pre_asterisk_qualifiers=None,
                 post_asterisk_qualifiers=None):
        """Create a new CL data type container.

        The CL type can either be a CL native type (``half``, ``double``, ``int``, ...) or the
        special ``mot_float_type`` type.

        Args:
            raw_data_type (str): the specific data type without the vector number and asterisks
            is_pointer_type (boolean): If this parameter is a pointer type (appended by a ``*``)
            vector_length (int or None): If None this data type is not a CL vector type.
                If it is an integer it is the vector length of this data type (2, 3, 4, ...)
            address_space_qualifier (str or None): the address space qualifier or None if not used. One of:
                {``__local``, ``local``, ``__global``, ``global``,
                ``__constant``, ``constant``, ``__private``, ``private``} or None.
            pre_asterisk_qualifiers (list of str or None): the type qualifiers to use before the (optional) asterisk.
                One or more of {const, volatile}
            post_asterisk_qualifiers (list of str or None): the type qualifiers to use after the (optional) asterisk.
                One or more of {const, restrict}
        """
        self._raw_data_type = str(raw_data_type)
        self._is_pointer_type = is_pointer_type
        self._vector_length = vector_length

        if self.vector_length:
            self._vector_length = int(self.vector_length)

        self.address_space_qualifier = address_space_qualifier

        self.pre_asterisk_qualifiers = pre_asterisk_qualifiers
        if isinstance(self.pre_asterisk_qualifiers, six.string_types):
            self.pre_asterisk_qualifiers = [self.pre_asterisk_qualifiers]

        self.post_asterisk_qualifiers = post_asterisk_qualifiers
        if isinstance(self.post_asterisk_qualifiers, six.string_types):
            self.post_asterisk_qualifiers = [post_asterisk_qualifiers]

    @classmethod
    def from_string(cls, parameter_declaration):
        """Parse the parameter declaration into a CLDataType

        Args:
            parameter_declaration (str): the CL parameter declaration. Example: ``global const float4*`` const

        Returns:
            mot.cl_data_type.SimpleCLDataType: the CL data type for this parameter declaration
        """
        from mot.parsers.cl.CLDataTypeParser import parse
        return parse(parameter_declaration)

    def get_declaration(self):
        declaration = ''
        if self.address_space_qualifier:
            declaration += str(self.address_space_qualifier) + ' '
        if self.pre_asterisk_qualifiers:
            declaration += str(' '.join(self.pre_asterisk_qualifiers)) + ' '
        declaration += str(self.declaration_type)
        if self.post_asterisk_qualifiers:
            declaration += ' ' + str(' '.join(self.post_asterisk_qualifiers)) + ' '
        return declaration

    @property
    def declaration_type(self):
        s = self.raw_data_type

        if self.vector_length is not None:
            s += str(self.vector_length)

        if self.is_pointer_type:
            s += '*'

        return str(s)

    @property
    def is_vector_type(self):
        return self.vector_length is not None

    @property
    def is_pointer_type(self):
        return self._is_pointer_type

    @property
    def vector_length(self):
        return self._vector_length

    @property
    def raw_data_type(self):
        return self._raw_data_type

    @property
    def ctype(self):
        if self.is_vector_type:
            return self._raw_data_type + str(self.vector_length)
        return self._raw_data_type

    def __str__(self):
        return self.get_declaration()
