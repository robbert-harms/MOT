import re
from mot.cl_data_type import SimpleCLDataType
from mot.parsers.cl.CLDataType import CLDataTypeParser, CLDataTypeSemantics


class Semantics(CLDataTypeSemantics):

    def __init__(self):
        super(Semantics, self).__init__()
        self._raw_data_type = None
        self._is_pointer_type = False
        self._vector_length = None
        self._address_space_qualifier = None
        self._pre_data_type_type_qualifiers = None
        self._post_data_type_type_qualifier = None

    def result(self, ast):
        return SimpleCLDataType(
            self._raw_data_type,
            is_pointer_type=self._is_pointer_type,
            vector_length=self._vector_length,
            address_space_qualifier=self._address_space_qualifier,
            pre_data_type_type_qualifiers=self._pre_data_type_type_qualifiers,
            post_data_type_type_qualifier=self._post_data_type_type_qualifier)

    def expr(self, ast):
        return ast

    def data_type(self, ast):
        return ast

    def is_pointer(self, ast):
        self._is_pointer_type = True
        return ast

    def scalar_data_type(self, ast):
        self._raw_data_type = ast
        return ast

    def vector_data_type(self, ast):
        match = re.match(r'(char|uchar|short|ushort|int|uint|long|ulong|float|double|half|mot_float_type)(\d+)', ast)
        self._raw_data_type = match.group(1)
        self._vector_length = match.group(2)
        return ast

    def user_data_type(self, ast):
        self._raw_data_type = ast
        return ast

    def address_space_qualifier(self, ast):
        self._address_space_qualifier = ast
        return ast

    def pre_data_type_type_qualifiers(self, ast):
        self._pre_data_type_type_qualifiers = ast
        return ast

    def post_data_type_type_qualifier(self, ast):
        self._post_data_type_type_qualifier = ast
        return ast


def parse(parameter_declaration):
    """Parse the parameter declaration into a CLDataType

    Args:
        parameter_declaration (str): the CL parameter declaration. Example: const float4* const test

    Returns:
        mot.cl_data_type.SimpleCLDataType: the CL data type for this parameter declaration
    """
    parser = CLDataTypeParser(parseinfo=False)
    return parser.parse(parameter_declaration, rule_name='result', semantics=Semantics())
