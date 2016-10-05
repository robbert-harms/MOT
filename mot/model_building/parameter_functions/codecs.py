__author__ = 'robbert'
__date__ = "5/15/14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractCodec(object):

    def get_cl_decode_function(self, fname='decodeParameters'):
        """Get a CL function that can transform the parameters from encoded space to model space.

        The signature of the CL function is:

        .. code-block:: c

            void <fname>(const mot_float_type* x);

        Args:
            fname (str): The CL function name to use

        Returns:
            str: An OpenCL function that is used in the CL kernel to transform the parameters from encoded space to
                model space so they can be used as input to the model.
        """
        pass

    def get_cl_encode_function(self, fname='encodeParameters'):
        """Get a CL function that can transform the parameters from model space to an encoded space.

        The signature of the CL function is:

        .. code-block:: c

            void <fname>(const mot_float_type* x);

        Args:
            fname (str): The CL function name to use

        Returns:
            str: An OpenCL function that is used in the CL kernel to transform the parameters from model space to
                encoded space so they can be used as input to an CL routine.
        """
        pass

    def get_nmr_parameters(self):
        """Get the number of parameters that are encoded and decoded.

        Returns:
            int: the number of parameters that are used in the decoding and encoding.
        """
        pass


class IdentityCodec(AbstractCodec):

    def __init__(self, nmr_parameters):
        """Create an identity codec.

        Input == output for this codec.

        Args:
            nmr_parameters (int): The number of parameters for the codec.
        """
        super(IdentityCodec, self).__init__()
        self._nmr_parameters = nmr_parameters

    def get_cl_decode_function(self, fname='decodeParameters', parameter_offset=0):
        func = '''
            void ''' + fname + '''(mot_float_type* x){'''
        for i in range(self._nmr_parameters):
            func += "\n" + "\t"*4 + 'x[' + str(i + parameter_offset) + '] = x[' + str(i + parameter_offset) + '];'
        return func + '''
            }
        '''

    def get_cl_encode_function(self, fname='encodeParameters', parameter_offset=0):
        func = '''
            void ''' + fname + '''(mot_float_type* x){'''
        for i in range(self._nmr_parameters):
            func += "\n" + "\t"*4 + 'x[' + str(i + parameter_offset) + '] = x[' + str(i + parameter_offset) + '];'
        return func + '''
            }
        '''

    def get_nmr_parameters(self):
        return self._nmr_parameters


class CodecBuilder(AbstractCodec):

    def __init__(self, enc_trans_list, dec_trans_list):
        """Build a codec out of the list of encode and decode transformation functions.

        Args:
            enc_trans_list (tuple): a listing of the encoding transformations, which are called in the given order.
                Each format should be a string with the format option {0} which is meant for the name of the array.
                The following is an example input (with two functions)::

                    ("{0}[0] = sqrt({0}[0]);",
                     "{0}[1] = sqrt({0}[1]);")

            enc_trans_list (tuple): a listing of the decoding transformations, which are called in the given order.
                Each format should be a string with the format option {0} which is meant for the name of the array.
                Example::

                        {0}[0] = {0}[0] * {0}[0];
                        {0}[1] = {0}[1] * {0}[1];
        """
        super(CodecBuilder, self).__init__()
        self._nmr_parameters = len(enc_trans_list)
        self._enc_trans_list = enc_trans_list
        self._dec_trans_list = dec_trans_list

    def get_cl_decode_function(self, fname='decodeParameters', parameter_offset=0):
        func = '''
            void ''' + fname + '''(mot_float_type* x){'''
        for d in self._dec_trans_list:
            func += "\n" + "\t"*4 + d.format('x')
        return func + '''
            }
        '''

    def get_cl_encode_function(self, fname='encodeParameters', parameter_offset=0):
        func = '''
            void ''' + fname + '''(mot_float_type* x){'''
        for d in self._enc_trans_list:
            func += "\n" + "\t"*4 + d.format('x')
        return func + '''
            }
        '''

    def get_nmr_parameters(self):
        return self._nmr_parameters


class CompositeCodec(AbstractCodec):

    def __init__(self, codec_list):
        """Given an ordered tuple of codecs, build a composite codec with the right indices.

        The encode and decode function of this class are the combination of the encode and decode functions of the
        given codecs in the codec list.

        Suppose we have four parameters and two codecs, one for one parameter and one for three parameters. This codec
        will construct a codec function which uses the first codec for the first parameter and the second codec for
        the remaining parameters. If the codecs were reversed, the first three parameters would be handled by the
        first codec and the last by the second. Order of the codec matters.

        Args:
            codec_list (tuple of AbstractCodecs): A list of codecs, used in that order.
        """
        super(CompositeCodec, self).__init__()
        self._codec_list = codec_list

    def get_cl_decode_function(self, fname='decodeParameters'):
        func = ''
        param_offset = 0
        for i, f in enumerate(self._codec_list):
            func += f.get_cl_decode_function(fname=(fname + '_' + str(i)), parameter_offset=param_offset)
            param_offset += f.get_decoded_space_width()

        func += '''
            void ''' + fname + '''(mot_float_type* x){
        '''
        for i in range(len(self._codec_list)):
            func += fname + '_' + str(i) + '(x);' + "\n"
        func += '''
            }
        '''
        return func

    def get_cl_encode_function(self, fname='encodeParameters'):
        func = ''
        param_offset = 0
        for i, f in enumerate(self._codec_list):
            func += f.get_cl_encode_function(fname=(fname + '_' + str(i)), parameter_offset=param_offset)
            param_offset += f.get_decoded_space_width()
        func += '''
            void ''' + fname + '''(mot_float_type* x){
        '''
        for i in range(len(self._codec_list)):
            func += fname + '_' + str(i) + '(x);' + "\n"
        func += '''
            }
        '''
        return func

    def get_nmr_parameters(self):
        return sum([x.get_decoded_space_width() for x in self._codec_list])
