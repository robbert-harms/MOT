import tatsu

__author__ = 'Robbert Harms'
__date__ = "2015-03-21"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


_cl_data_type_parser = tatsu.compile('''
    result = [address_space] {pre_type_qualifiers}* data_type [{post_type_qualifiers}*];

    address_space = ['__'] ('local' | 'global' | 'constant' | 'private');
    pre_type_qualifiers = 'const' | 'volatile';
    
    data_type = type_specifier [vector_length] {array_size}* {pointer}*;
    pointer = '*';
    vector_length = /[2348(16]/;
    array_size = /\[\d+\]/;
    
    post_type_qualifiers = 'const' | 'restrict';
    
    type_specifier = ?'(unsigned )?\w[\w]*[a-zA-Z]';
''')


class CLDataType:
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

        For example, if the data type is float4*, we will return float4 here.

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
    def nmr_pointers(self):
        """Get the number of asterisks / pointer references of this data type.

        If the data type is float**, we return 2 here.

        Returns:
            int: the number of pointer asterisks in the data type.
        """
        raise NotImplementedError()

    @property
    def is_array_type(self):
        """Check if this parameter is an array type (like float[3] or int[10][5]).

        Returns:
            boolean: True if this is an array type, false otherwise
        """
        raise NotImplementedError()

    @property
    def array_sizes(self):
        """Get the dimension of this array type.

        This returns for example (10, 5) for the data type float[10][5].

        Returns:
            Tuple[int]: the sizes of the arrays
        """
        raise NotImplementedError()

    @property
    def vector_length(self):
        """Get the length of this vector, returns None if not a vector type.

        Returns:
            int: the length of the vector type (for example, if the data type is float4, this returns 4).
        """
        raise NotImplementedError()

    @property
    def address_space(self):
        """Get the address space of this data declaration.

        Returns:
            str: the data type address space, one of ``global``, ``local``, ``constant`` or ``private``.
        """
        raise NotImplementedError()


class SimpleCLDataType(CLDataType):

    def __init__(self, raw_data_type, nmr_pointers=0, array_sizes=None, vector_length=None, address_space=None,
                 pre_type_qualifiers=None, post_type_qualifiers=None):
        """Create a new CL data type container.

        The CL type can either be a CL native type (``half``, ``double``, ``int``, ...) or the
        special ``mot_float_type`` type.

        Args:
            raw_data_type (str): the specific data type without the vector number and asterisks
            nmr_pointers (int): The number of dereferences of this parameter (number of ``*``).
            array_sizes (List[Int]): if this parameter is supposed to be an array (i.e. float[10] or int[10][3]),
                this array lists those sizes
            vector_length (int or None): If None this data type is not a CL vector type.
                If it is an integer it is the vector length of this data type (2, 3, 4, ...)
            address_space (str or None): the address space qualifier or None if not used. One of:
                {``__local``, ``local``, ``__global``, ``global``,
                ``__constant``, ``constant``, ``__private``, ``private``} or None.
            pre_type_qualifiers (Union[List[str], None]): the type qualifiers to use before the type.
                One or more of {const, volatile}
            post_type_qualifiers (Union[List[str], None]): the type qualifiers to use before the type.
                One or more of {const, restrict}
        """
        self._raw_data_type = str(raw_data_type)
        self._nmr_pointers = nmr_pointers
        self._vector_length = vector_length
        self._array_sizes = array_sizes or ()

        if self.vector_length:
            self._vector_length = int(self.vector_length)

        self._address_space = address_space

        if self._address_space is not None:
            if '__' in self._address_space:
                self._address_space = self._address_space[2:]

            valid_address_spaces = ('global', 'constant', 'local', 'private')

            if self._address_space not in valid_address_spaces:
                raise ValueError('The given address space qualifier "{}" is not one of {}.'.format(
                    self._address_space, valid_address_spaces))

        self.pre_type_qualifiers = pre_type_qualifiers
        if isinstance(self.pre_type_qualifiers, str):
            self.pre_type_qualifiers = [self.pre_type_qualifiers]

        self.post_type_qualifiers = post_type_qualifiers
        if isinstance(self.post_type_qualifiers, str):
            self.post_type_qualifiers = [post_type_qualifiers]

    @classmethod
    def from_string(cls, parameter_declaration):
        """Parse the parameter declaration into a CLDataType

        Args:
            parameter_declaration (str): the CL parameter declaration. Example: ``global const float4*`` const

        Returns:
            SimpleCLDataType: the CL data type for this parameter declaration
        """
        class Semantics:

            def __init__(self):
                self._raw_data_type = None
                self._nmr_pointers = 0
                self._vector_length = None
                self._address_space = None
                self._pre_type_qualifiers = None
                self._post_type_qualifiers = None
                self._array_sizes = []

            def result(self, ast):
                return SimpleCLDataType(
                    self._raw_data_type,
                    nmr_pointers=self._nmr_pointers,
                    array_sizes=self._array_sizes,
                    vector_length=self._vector_length,
                    address_space=self._address_space,
                    pre_type_qualifiers=self._pre_type_qualifiers,
                    post_type_qualifiers=self._post_type_qualifiers)

            def pointer(self, ast):
                self._nmr_pointers += 1
                return ast

            def vector_length(self, ast):
                self._vector_length = int(ast)
                return ast

            def type_specifier(self, ast):
                self._raw_data_type = ast
                return ast

            def address_space(self, ast):
                self._address_space = ''.join(ast)
                return ast

            def pre_type_qualifiers(self, ast):
                self._pre_type_qualifiers = ast
                return ast

            def post_type_qualifiers(self, ast):
                self._post_type_qualifiers = ast
                return ast

            def array_size(self, ast):
                self._array_sizes.append(int(ast[1:-1]))
                return ast

        return _cl_data_type_parser.parse(parameter_declaration, semantics=Semantics())

    def get_declaration(self):
        declaration = ''
        if self._address_space:
            declaration += str(self._address_space) + ' '
        if self.pre_type_qualifiers:
            declaration += str(' '.join(self.pre_type_qualifiers)) + ' '
        declaration += str(self.declaration_type)
        if self.post_type_qualifiers:
            declaration += ' ' + str(' '.join(self.post_type_qualifiers)) + ' '
        return declaration

    @property
    def declaration_type(self):
        s = self.raw_data_type

        if self.vector_length is not None:
            s += str(self.vector_length)

        if self.is_pointer_type:
            s += '*' * self._nmr_pointers

        if self.is_array_type:
            s += ''.join('[{}]'.format(s) for s in self._array_sizes)

        return str(s)

    @property
    def is_vector_type(self):
        return self.vector_length is not None

    @property
    def is_pointer_type(self):
        return self._nmr_pointers > 0

    @property
    def nmr_pointers(self):
        return self._nmr_pointers

    @property
    def is_array_type(self):
        return len(self._array_sizes)

    @property
    def array_sizes(self):
        return self._array_sizes

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

    @property
    def address_space(self):
        if self._address_space is None:
            return 'private'
        return self._address_space

    def __str__(self):
        return self.get_declaration()
