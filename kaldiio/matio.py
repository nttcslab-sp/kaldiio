from functools import partial
import os
import re
import struct
import sys

import numpy as np
from six import binary_type
from six import BytesIO
from six.moves import cStringIO as StringIO
from six import string_types

from kaldiio.compression_header import GlobalHeader
from kaldiio.compression_header import PerColHeader
from kaldiio.utils import convert_to_slice
from kaldiio.utils import MultiFileDescriptor
from kaldiio.utils import open_like_kaldi
from kaldiio.utils import open_or_fd

PY3 = sys.version_info[0] == 3

if PY3:
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping


def load_scp(fname, endian='<', separator=' ', as_bytes=False):
    """Lazy loader for kaldi scp file.

    Args:
        fname (str or file(text mode)):
        endian (str):
        separator (str):
        as_bytes (bool): Read as raw bytes string
    """
    loader = LazyLoader(partial(load_mat, endian=endian, as_bytes=as_bytes))
    with open_or_fd(fname, 'r') as fd:
        for line in fd:
            try:
                token, arkname = line.split(separator, 1)
            except ValueError as e:
                raise ValueError(
                    str(e) + '\nFile format is wrong?')

            loader[token] = arkname
    return loader


def load_mat(ark_name, endian='<', as_bytes=False):
    slices = None
    if ':' in ark_name:
        fname, offset = ark_name.split(':', 1)
        if '[' in offset and ']' in offset:
            offset, Range = offset.split('[')
            slices = convert_to_slice(Range.replace(']', '').strip())
        offset = int(offset)
    else:
        fname = ark_name
        offset = None

    with open_like_kaldi(fname, 'rb') as fd:
        if offset is not None:
            fd.seek(offset)
        if not as_bytes:
            array = read_kaldi(fd, endian)
        else:
            array = fd.read()
    if slices is not None:
        array = array[slices]
    return array


class LazyLoader(MutableMapping):
    """Don't use this class directly"""
    def __init__(self, loader):
        self._dict = {}
        self._loader = loader

    def __repr__(self):
        return 'LazyLoader [{} keys]'.format(len(self))

    def __getitem__(self, key):
        ark_name = self._dict[key]
        try:
            return self._loader(ark_name)
        except Exception:
            sys.stderr.write(
                'Error at loading "{}"'.format(ark_name))
            raise

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return len(self._dict)

    def __contains__(self, item):
        return item in self._dict


def load_ark(fname, return_position=False, endian='<'):
    size = 0
    with open_or_fd(fname, 'rb') as fd:
        while True:
            token = read_token(fd)
            if token is None:
                break
            size += len(token) + 1
            array, _size = read_kaldi(fd, endian, return_size=True)
            if return_position:
                yield token, array, size
            else:
                yield token, array
            size += _size


def read_token(fd):
    """Read token

    Args:
        fd (file):
    """
    token = []
    while True:
        char = fd.read(1)
        if isinstance(char, binary_type):
            char = char.decode()
        if char == ' ' or char == '':
            break
        else:
            token.append(char)
    if len(token) == 0:  # End of file
        return None
    return ''.join(token)


def read_kaldi(fd, endian='<', return_size=False):
    """Load kaldi

    Args:
        fd (file): Binary mode file object. Cannot input string
        endian (str):
        return_size (bool):
    """
    binary_flag = fd.read(2)
    assert isinstance(binary_flag, binary_type)
    fd = MultiFileDescriptor(BytesIO(binary_flag), fd)
    if binary_flag != b'\0B':  # Load as ascii
        array, size = read_ascii_mat(fd, return_size=True)

    else:  # Load as binary
        # Check binary type
        head = fd.read(3)
        fd = MultiFileDescriptor(BytesIO(head), fd)
        if b'\0B\4' == head:  # This is int32Vector
            array, size = read_int32vector(fd, endian, return_size=True)
        else:
            array, size = read_matrix_or_vector(fd, endian, return_size=True)

    if return_size:
        return array, size
    else:
        return array


def read_int32vector(fd, endian='<', return_size=False):
    assert fd.read(2) == b'\0B'
    assert fd.read(1) == b'\4'
    length = struct.unpack(endian + 'i', fd.read(4))[0]
    array = np.empty(length, dtype=np.int32)
    for i in range(length):
        assert fd.read(1) == b'\4'
        array[i] = struct.unpack(endian + 'i', fd.read(4))[0]
    if return_size:
        return array, (length + 1) * 5 + 2
    else:
        return array


def read_matrix_or_vector(fd, endian='<', return_size=False):
    """Call from load_kaldi_file

    Args:
        fd (file):
        endian (str):
        return_size (bool):
    """
    size = 0
    assert fd.read(2) == b'\0B'
    size += 2

    Type = str(read_token(fd))
    size += len(Type) + 1

    # CompressedMatrix
    if 'CM' == Type:
        # Read GlobalHeader
        global_header = GlobalHeader.read(fd, Type, endian)
        size += global_header.size
        per_col_header = PerColHeader.read(fd, global_header)
        size += per_col_header.size

        # Read data
        buf = fd.read(global_header.rows * global_header.cols)
        size += global_header.rows * global_header.cols
        array = np.frombuffer(buf, dtype=np.dtype(endian + 'u1'))
        array = array.reshape((global_header.cols, global_header.rows))

        # Decompress
        array = per_col_header.char_to_float(array)
        array = array.T

    elif 'CM2' == Type:
        # Read GlobalHeader
        global_header = GlobalHeader.read(fd, Type, endian)
        size += global_header.size

        # Read matrix
        buf = fd.read(2 * global_header.rows * global_header.cols)
        array = np.frombuffer(buf, dtype=np.dtype(endian + 'u2'))
        array = array.reshape((global_header.rows, global_header.cols))

        # Decompress
        array = global_header.uint_to_float(array)

    elif 'CM3' == Type:
        # Read GlobalHeader
        global_header = GlobalHeader.read(fd, Type, endian)
        size += global_header.size

        # Read matrix
        buf = fd.read(global_header.rows * global_header.cols)
        array = np.frombuffer(buf, dtype=np.dtype(endian + 'u1'))
        array = array.reshape((global_header.rows, global_header.cols))

        # Decompress
        array = global_header.uint_to_float(array)

    else:
        if Type == 'FM' or Type == 'FV':
            dtype = endian + 'f'
            bytes_per_sample = 4
        elif Type == 'DM' or Type == 'DV':
            dtype = endian + 'd'
            bytes_per_sample = 8
        else:
            raise ValueError(
                'Unexpected format: "{}". Now FM, FV, DM, DV, '
                'CM, CM2, CM3 are supported.'.format(Type))

        assert fd.read(1) == b'\4'
        size += 1
        rows = struct.unpack(endian + 'i', fd.read(4))[0]
        size += 4
        dim = rows
        if 'M' in Type:  # As matrix
            assert fd.read(1) == b'\4'
            size += 1
            cols = struct.unpack(endian + 'i', fd.read(4))[0]
            size += 4
            dim = rows * cols

        buf = fd.read(dim * bytes_per_sample)
        size += dim * bytes_per_sample
        array = np.frombuffer(buf, dtype=np.dtype(dtype))

        if 'M' in Type:  # As matrix
            array = np.reshape(array, (rows, cols))

    if return_size:
        return array, size
    else:
        return array


def read_ascii_mat(fd, return_size=False):
    """Call from load_kaldi_file

    Args:
        fd (file): binary mode
        return_size (bool):
    """
    string = []
    size = 0

    # Find '[' char
    while True:
        try:
            char = fd.read(1).decode()
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                str(e) + '\nFile format is wrong?')
        size += 1
        if char == ' ' or char == os.linesep:
            continue
        elif char == '[':
            hasparent = True
            break
        else:
            string.append(char)
            hasparent = False
            break

    # Read data
    ndmin = 1
    while True:
        char = fd.read(1).decode()
        size += 1
        if hasparent:
            if char == ']':
                char = fd.read(1).decode()
                size += 1
                assert char == os.linesep or char == ''
                break
            elif char == os.linesep:
                ndmin = 2
            elif char == '':
                raise ValueError(
                    'There are no corresponding bracket \']\' with \'[\'')
        else:
            if char == os.linesep or char == '':
                break
        string.append(char)
    string = ''.join(string)
    assert len(string) != 0

    # Examine dtype
    match = re.match(r' *([^ \n]+) *', string)
    if match is None:
        dtype = np.float32
    else:
        ma = match.group(0)
        # If first element is integer, deal as interger array
        try:
            float(ma)
        except ValueError:
            raise RuntimeError(
                ma + 'is not a digit\nFile format is wrong?')
        if '.' in ma:
            dtype = np.float32
        else:
            dtype = np.int32
    array = np.loadtxt(StringIO(string), dtype=dtype, ndmin=ndmin)
    if return_size:
        return array, size
    else:
        return array


def _get_offset(line, separator=' ', endian='<', as_bytes=False):
    token, ark_name = line.split(separator, 1)
    if ':' in ark_name:
        fname, offset = ark_name.split(':', 1)
        if '[' in offset and ']' in offset:
            offset, Range = offset.split('[')
        offset = int(offset)
    else:
        fname = ark_name
        offset = None

    with open_like_kaldi(fname, 'rb') as fd:
        if offset is not None:
            fd.seek(offset)
        if not as_bytes:
            array, _size = read_kaldi(fd, endian, return_size=True)
        else:
            array = fd.read()
            _size = len(array)
    return offset + _size


def save_ark(ark, array_dict, scp=None,
             append=False, text=False,
             as_bytes=False,
             endian='<',
             compression_method=None):
    """Write ark

    Args:
        ark (str or fd):
        array_dict (dict):
        scp (str or fd):
        append (bool): If True is specified, open the file
            with appendable mode
        text (bool): If True, saving in text ark format.
        as_bytes (bool): Save the value of the input array_dict as just a
            bytes string.
        endian (str):
        compression_method (int):
    """
    if isinstance(ark, string_types):
        seekable = True
    # Maybe, never match with this
    elif not hasattr(ark, 'tell'):
        seekable = False
    else:
        try:
            ark.tell()
            seekable = True
        except Exception:
            seekable = False

    if scp is not None and not isinstance(ark, string_types):
        if not seekable:
            raise TypeError('scp file can be created only '
                            'if the output ark file is a file or '
                            'a seekable file descriptor.')

    # Write ark
    mode = 'ab' if append else 'wb'
    pos_list = []
    with open_or_fd(ark, mode) as fd:
        if seekable:
            offset = fd.tell()
        else:
            offset = 0
        size = 0
        for key in array_dict:
            fd.write((key + ' ').encode())
            size += len(key) + 1
            pos_list.append(size)
            if as_bytes:
                byte = bytes(array_dict[key])
                size += len(byte)
                fd.write(byte)
            else:
                if text:
                    size += write_array_ascii(fd, array_dict[key], endian)
                else:
                    size += write_array(fd, array_dict[key], endian,
                                        compression_method)

    # Write scp
    mode = 'a' if append else 'w'
    if scp is not None:
        name = ark if isinstance(ark, string_types) else ark.name
        with open_or_fd(scp, mode) as fd:
            for key, position in zip(array_dict, pos_list):
                fd.write(key + ' ' + name + ':' +
                         str(position + offset) + os.linesep)


def save_mat(fname, array, endian='<', compression_method=None):
    with open_or_fd(fname, 'rb') as fd:
        return write_array(fd, array, endian, compression_method)


def write_array(fd, array, endian='<', compression_method=None):
    """Write array

    Args:
        fd (file): binary mode
        array (np.ndarray):
        endian (str):
    Returns:
        size (int):
    """
    size = 0
    assert isinstance(array, np.ndarray)
    fd.write(b'\0B')
    size += 2
    if endian not in array.dtype.str:
        array = array.astype(array.dtype.newbyteorder())
    if compression_method is not None:
        if array.ndim != 2:
            raise ValueError(
                'array must be matrix if compression_method is not None: {}'
                .format(array.ndim))

        global_header = GlobalHeader.compute(array, compression_method)
        size += global_header.write(fd, endian)
        if global_header.type == 'CM':
            per_col_header = PerColHeader.compute(array, global_header)
            size += per_col_header.write(fd, global_header)

            array = per_col_header.float_to_char(array.T)

            byte_string = array.tobytes()
            fd.write(byte_string)
            size += len(byte_string)

        elif global_header.type == 'CM2':
            array = global_header.float_to_uint(array)

            byte_string = array.tobytes()
            fd.write(byte_string)
            size += len(byte_string)

        elif global_header.type == 'CM3':
            array = global_header.float_to_uint(array)

            byte_string = array.tobytes()
            fd.write(byte_string)
            size += len(byte_string)

    elif array.dtype == np.int32:
        assert len(array.shape) == 1  # Must be vector
        fd.write(b'\4')
        fd.write(struct.pack(endian + 'i', len(array)))
        for x in array:
            fd.write(b'\4')
            fd.write(struct.pack(endian + 'i', x))
        size += (len(array) + 1) * 5

    elif array.dtype == np.float32 or array.dtype == np.float64:
        assert 0 < len(array.shape) < 3  # Matrix or vector
        if len(array.shape) == 1:
            if array.dtype == np.float32:
                fd.write(b'FV ')
                size += 3
            elif array.dtype == np.float64:
                fd.write(b'DV ')
                size += 3
            fd.write(b'\4')
            size += 1
            fd.write(struct.pack(endian + 'i', len(array)))
            size += 4

        elif len(array.shape) == 2:
            if array.dtype == np.float32:
                fd.write(b'FM ')
                size += 3
            elif array.dtype == np.float64:
                fd.write(b'DM ')
                size += 3
            fd.write(b'\4')
            size += 1
            fd.write(struct.pack(endian + 'i', len(array)))  # Rows
            size += 4

            fd.write(b'\4')
            size += 1
            fd.write(struct.pack(endian + 'i', array.shape[1]))  # Cols
            size += 4
        fd.write(array.tobytes())
        size += array.nbytes
    else:
        raise ValueError('Unsupported array type: {}'.format(array.dtype))
    return size


def write_array_ascii(fd, array, digit='.12g'):
    """write_array_ascii

    Args:
        fd (file): binary mode
        array (np.ndarray):
        digit (str):
    Returns:
        size (int):
    """
    assert isinstance(array, np.ndarray)
    assert array.ndim in (1, 2)
    size = 0
    fd.write(b' [')
    size += 2
    if array.ndim == 2:
        for row in array:
            fd.write(b'\n  ')
            size += 3
            for i in row:
                string = format(i, digit)
                fd.write(string.encode())
                fd.write(b' ')
                size += len(string) + 1
        fd.write(b']\n')
        size += 2
    elif array.ndim == 1:
        fd.write(b' ')
        size += 1
        for i in array:
            string = format(i, digit)
            fd.write(string.encode())
            fd.write(b' ')
            size += len(string) + 1
        fd.write(b']\n')
        size += 2
    return size
