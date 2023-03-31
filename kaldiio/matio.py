from __future__ import division
from __future__ import unicode_literals

import codecs
import math
import pickle
import re
import struct
import sys
import warnings
from functools import partial
from io import BytesIO
from io import StringIO

import numpy as np

from kaldiio.compression_header import GlobalHeader
from kaldiio.compression_header import PerColHeader
from kaldiio.utils import LazyLoader
from kaldiio.utils import LimitedSizeDict
from kaldiio.utils import MultiFileDescriptor
from kaldiio.utils import default_encoding
from kaldiio.utils import open_like_kaldi
from kaldiio.utils import open_or_fd
from kaldiio.utils import seekable
from kaldiio.wavio import read_wav
from kaldiio.wavio import write_wav

PY3 = sys.version_info[0] == 3

if PY3:
    from collections.abc import Mapping

    binary_type = bytes
    string_types = (str,)

    def to_bytes(n, length, endianess="little"):
        return n.to_bytes(length, endianess)

    def from_bytes(s, endianess="little"):
        return int.from_bytes(s, endianess)

else:
    from collections import Mapping

    binary_type = str
    string_types = (basestring,)  # noqa: F821

    def to_bytes(n, length, endianess="little"):
        assert endianess in ("big", "little"), endianess
        h = b"%x" % n
        s = codecs.decode((b"0" * (len(h) % 2) + h).zfill(length * 2), "hex")
        return s if endianess == "big" else s[::-1]

    def from_bytes(s, endianess="little"):
        if endianess == "little":
            s = s[::-1]
        return int(codecs.encode(s, "hex"), 16)


def load_scp(fname, endian="<", separator=None, segments=None, max_cache_fd=0):
    """Lazy loader for kaldi scp file.

    Args:
        fname (str or file(text mode)):
        endian (str):
        separator (str):
        segments (str): The path of segments
    """
    assert endian in ("<", ">"), endian

    if max_cache_fd != 0:
        if segments is not None:
            raise ValueError("max_cache_fd is not supported for segments mode")
        d = LimitedSizeDict(max_cache_fd)
    else:
        d = None

    if segments is None:
        load_func = partial(load_mat, endian=endian, fd_dict=d)
        loader = LazyLoader(load_func)
        with open_like_kaldi(fname, "r") as fd:
            for line in fd:
                seps = line.split(separator, 1)
                if len(seps) != 2:
                    raise ValueError("Invalid line is found:\n>   {}".format(line))
                token, arkname = seps
                loader[token] = arkname.rstrip()
        return loader
    else:
        return SegmentsExtractor(fname, separator=separator, segments=segments)


def load_scp_sequential(fname, endian="<", separator=None, segments=None):
    """Lazy loader for kaldi scp file.

    Args:
        fname (str or file(text mode)):
        endian (str):
        separator (str):
        segments (str): The path of segments
    """
    assert endian in ("<", ">"), endian
    if segments is None:
        with open_like_kaldi(fname, "r") as fd:
            prev_ark = None
            prev_arkfd = None

            try:
                for line in fd:
                    seps = line.split(separator, 1)
                    if len(seps) != 2:
                        raise ValueError("Invalid line is found:\n>   {}".format(line))
                    token, arkname = seps
                    arkname = arkname.rstrip()

                    ark, offset, slices = _parse_arkpath(arkname)

                    if prev_ark == ark:
                        arkfd = prev_arkfd
                        mat = _load_mat(arkfd, offset, slices, endian=endian)
                    else:
                        if prev_arkfd is not None:
                            prev_arkfd.close()
                        arkfd = open_like_kaldi(ark, "rb")
                        mat = _load_mat(arkfd, offset, slices, endian=endian)

                    prev_ark = ark
                    prev_arkfd = arkfd
                    yield token, mat
            except Exception:
                if prev_arkfd is not None:
                    prev_arkfd.close()
                raise

    else:
        for data in SegmentsExtractor(
            fname, separator=separator, segments=segments
        ).generator():
            yield data


def load_wav_scp(fname, segments=None, separator=None):
    warnings.warn("Use load_scp instead of load_wav_scp", DeprecationWarning)
    return load_scp(fname, separator=separator, segments=segments)


class SegmentsExtractor(Mapping):
    """Emulate the following,

    https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/extract-segments.cc

    Args:
        segments (str): The file format is
            "<segment-id> <recording-id> <start-time> <end-time>\n"
            "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5\n"
    """

    def __init__(self, fname, segments=None, separator=None):
        self.wav_scp = fname
        self.wav_loader = load_scp(self.wav_scp, separator=separator)

        self.segments = segments
        self._segments_dict = {}
        with open_or_fd(self.segments, "r") as f:
            for line in f:
                sps = line.rstrip().split(separator)
                if len(sps) != 4:
                    raise RuntimeError("Format is invalid: {}".format(line))
                uttid, recodeid, st, et = sps
                self._segments_dict[uttid] = (recodeid, float(st), float(et))

                if recodeid not in self.wav_loader:
                    raise RuntimeError(
                        'Not found "{}" in {}'.format(recodeid, self.wav_scp)
                    )

    def generator(self):
        recodeid_counter = {}
        for utt, (recodeid, st, et) in self._segments_dict.items():
            recodeid_counter[recodeid] = recodeid_counter.get(recodeid, 0) + 1

        cached = {}
        for utt, (recodeid, st, et) in self._segments_dict.items():
            if recodeid not in cached:
                cached[recodeid] = self.wav_loader[recodeid]
            array = cached[recodeid]

            # Keep array until the last query
            recodeid_counter[recodeid] -= 1
            if recodeid_counter[recodeid] == 0:
                cached.pop(recodeid)

            yield utt, self._return(array, st, et)

    def __iter__(self):
        return iter(self._segments_dict)

    def __contains__(self, item):
        return item in self._segments_dict

    def __len__(self):
        return len(self._segments_dict)

    def __getitem__(self, key):
        recodeid, st, et = self._segments_dict[key]
        array = self.wav_loader[recodeid]
        return self._return(array, st, et)

    def _return(self, array, st, et):
        if isinstance(array, (tuple, list)):
            rate, array = array
        else:
            raise RuntimeError("{} is not wav.scp?".format(self.wav_scp))

        # Convert starting time of the segment to corresponding sample number.
        # If end time is -1 then use the whole file starting from start time.
        if et != -1:
            return rate, array[int(st * rate) : int(et * rate)]
        else:
            return rate, array[int(st * rate) :]


def load_mat(ark_name, endian="<", fd_dict=None):
    assert endian in ("<", ">"), endian
    if fd_dict is not None and not isinstance(fd_dict, Mapping):
        raise RuntimeError(
            "fd_dict must be dict or None, bot got {}".format(type(fd_dict))
        )

    ark, offset, slices = _parse_arkpath(ark_name)

    if fd_dict is not None and not (ark.strip()[-1] == "|" or ark.strip()[0] == "|"):
        if ark not in fd_dict:
            fd_dict[ark] = open_like_kaldi(ark, "rb")
        fd = fd_dict[ark]
        return _load_mat(fd, offset, slices, endian=endian)
    else:
        with open_like_kaldi(ark, "rb") as fd:
            return _load_mat(fd, offset, slices, endian=endian)


def _parse_arkpath(ark_name):
    """Parse arkpath

    Args:
        ark_name (str):
    Returns:
        Tuple[str, int, Optional[Tuple[slice, ...]]]
    Examples:
        >>> _parse_arkpath('a.ark')
        'a.ark', None, None
        >>> _parse_arkpath('a.ark:12')
        'a.ark', 12, None
        >>> _parse_arkpath('a.ark:12[3:4]')
        'a.ark', 12, (slice(3, 4, None),)
        >>> _parse_arkpath('cat "fo:o.ark" |')
        'cat "fo:o.ark" |', None, None

    """

    if ark_name.strip()[-1] == "|" or ark_name.strip()[0] == "|":
        # Something like: "| cat foo" or "cat bar|" shouldn't be parsed
        return ark_name, None, None

    slices = None
    if "[" in ark_name and "]" in ark_name:
        _ark_name, Range = ark_name.split("[")
        Range = Range.replace("]", "").strip()
        try:
            slices = _convert_to_slice(Range)
        except Exception:
            pass
        else:
            ark_name = _ark_name

    if ":" in ark_name:
        fname, offset = ark_name.rsplit(":", 1)
        try:
            offset = int(offset)
        except ValueError:
            fname = ark_name
            offset = None
    else:
        fname = ark_name
        offset = None
    return fname, offset, slices


def _convert_to_slice(string):
    """Convert slice-str to slice

    Examples:
        >>> _convert_to_slice('0:51')
        (slice(0, 52),)
        >>> _convert_to_slice('0:51,6:10')
        (slice(0, 52), slice(6, 11))
        >>> _convert_to_slice(',6:10')
        (slice(None), slice(6, 11))

    """

    slices = []
    for ele in string.split(","):
        if ele == "" or ele == ":":
            slices.append(slice(None))
        else:
            sps = []
            for sp in ele.split(":"):
                try:
                    sps.append(int(sp))
                except ValueError:
                    raise ValueError("Format error: {}".format(string))
            if len(sps) == 1:
                sl = slice(sps[0], sps[0] + 1)
            elif len(sps) == 2:
                sl = slice(sps[0], sps[1] + 1)
            elif len(sps) == 3:
                sl = slice(sps[0], sps[1] + 1, sps[2])
            else:
                raise RuntimeError("Too many : {}".format(string))

            slices.append(sl)
    return tuple(slices)


def _load_mat(fd, offset, slices=None, endian="<"):
    if offset is not None:
        fd.seek(offset)
    array = read_kaldi(fd, endian)

    if slices is not None:
        if isinstance(array, (tuple, list)):
            array = (array[0], array[1][slices])
        else:
            array = array[slices]
    return array


def load_ark(fname, endian="<"):
    assert endian in ("<", ">"), endian
    with open_or_fd(fname, "rb") as fd:
        while True:
            token = read_token(fd)
            if token is None:
                break
            array = read_kaldi(fd, endian)
            yield token, array


def read_token(fd):
    """Read token

    Args:
        fd (file):
    """
    token = []
    # Keep the loop until finding ' ' or end of char
    while True:
        c = fd.read(1)
        if c == b" " or c == b"":
            break
        token.append(c)
    if len(token) == 0:  # End of file
        return None
    decoded = b"".join(token).decode(encoding=default_encoding)
    return decoded


def read_kaldi(fd, endian="<", audio_loader="soundfile", load_kwargs=None):
    """Load kaldi

    Args:
        fd (file): Binary mode file object. Cannot input string
        endian (str):
        audio_loader: (Union[str, callable]):
    """
    assert endian in ("<", ">"), endian
    if load_kwargs is None:
        load_kwargs = {}

    max_flag_length = len(b"AUDIO")

    binary_flag = fd.read(max_flag_length)
    assert isinstance(binary_flag, binary_type), type(binary_flag)

    if seekable(fd):
        fd.seek(-max_flag_length, 1)
    else:
        fd = MultiFileDescriptor(BytesIO(binary_flag), fd)

    if binary_flag[:4] == b"RIFF":
        # array: Tuple[int, np.ndarray]
        array = read_wav(fd)

    elif binary_flag[:3] == b"NPY":
        fd.read(3)
        length_ = _read_length_header(fd)
        buf = fd.read(length_)
        _fd = BytesIO(buf)
        array = np.load(_fd, **load_kwargs)

    elif binary_flag[:3] == b"PKL":
        fd.read(3)
        array = pickle.load(fd, **load_kwargs)

    elif binary_flag[:5] == b"AUDIO":
        fd.read(5)
        length_ = _read_length_header(fd)
        buf = fd.read(length_)
        _fd = BytesIO(buf)

        if audio_loader == "soundfile":
            import soundfile

            audio_loader = soundfile.read
        else:
            raise ValueError("Not supported: audio_loader={}".format(audio_loader))

        x1, x2 = audio_loader(_fd, **load_kwargs)

        # array: Tuple[int, np.ndarray] according to scipy wav read
        if isinstance(x1, int) and isinstance(x2, np.ndarray):
            array = (x1, x2)
        elif isinstance(x1, np.ndarray) and isinstance(x2, int):
            array = (x2, x1)
        else:
            raise RuntimeError(
                "Got unexpected type from audio_loader: ({}, {})".format(
                    type(x1), type(x2)
                )
            )

    # Load as binary
    elif binary_flag[:2] == b"\0B":
        if binary_flag[2:3] == b"\4":  # This is int32Vector
            array = read_int32vector(fd, endian)
        else:
            array = read_matrix_or_vector(fd, endian)
    # Load as ascii
    else:
        array = read_ascii_mat(fd)

    return array


def read_int32vector(fd, endian="<", return_size=False):
    assert fd.read(2) == b"\0B"
    assert fd.read(1) == b"\4"
    length = struct.unpack(endian + "i", fd.read(4))[0]
    array = np.empty(length, dtype=np.int32)
    for i in range(length):
        assert fd.read(1) == b"\4"
        array[i] = struct.unpack(endian + "i", fd.read(4))[0]
    if return_size:
        return array, (length + 1) * 5 + 2
    else:
        return array


def read_matrix_or_vector(fd, endian="<", return_size=False):
    """Call from load_kaldi_file

    Args:
        fd (file):
        endian (str):
        return_size (bool):
    """
    size = 0
    assert fd.read(2) == b"\0B"
    size += 2

    Type = str(read_token(fd))
    size += len(Type) + 1

    # CompressedMatrix
    if "CM" == Type:
        # Read GlobalHeader
        global_header = GlobalHeader.read(fd, Type, endian)
        size += global_header.size
        per_col_header = PerColHeader.read(fd, global_header)
        size += per_col_header.size

        # Read data
        buf = fd.read(global_header.rows * global_header.cols)
        size += global_header.rows * global_header.cols
        array = np.frombuffer(buf, dtype=np.dtype(endian + "u1"))
        array = array.reshape((global_header.cols, global_header.rows))

        # Decompress
        array = per_col_header.char_to_float(array)
        array = array.T

    elif "CM2" == Type:
        # Read GlobalHeader
        global_header = GlobalHeader.read(fd, Type, endian)
        size += global_header.size

        # Read matrix
        buf = fd.read(2 * global_header.rows * global_header.cols)
        array = np.frombuffer(buf, dtype=np.dtype(endian + "u2"))
        array = array.reshape((global_header.rows, global_header.cols))

        # Decompress
        array = global_header.uint_to_float(array)

    elif "CM3" == Type:
        # Read GlobalHeader
        global_header = GlobalHeader.read(fd, Type, endian)
        size += global_header.size

        # Read matrix
        buf = fd.read(global_header.rows * global_header.cols)
        array = np.frombuffer(buf, dtype=np.dtype(endian + "u1"))
        array = array.reshape((global_header.rows, global_header.cols))

        # Decompress
        array = global_header.uint_to_float(array)

    else:
        if Type == "FM" or Type == "FV":
            dtype = endian + "f"
            bytes_per_sample = 4
        elif Type == "DM" or Type == "DV":
            dtype = endian + "d"
            bytes_per_sample = 8
        else:
            raise ValueError(
                'Unexpected format: "{}". Now FM, FV, DM, DV, '
                "CM, CM2, CM3 are supported.".format(Type)
            )

        assert fd.read(1) == b"\4"
        size += 1
        rows = struct.unpack(endian + "i", fd.read(4))[0]
        size += 4
        dim = rows
        if "M" in Type:  # As matrix
            assert fd.read(1) == b"\4"
            size += 1
            cols = struct.unpack(endian + "i", fd.read(4))[0]
            size += 4
            dim = rows * cols

        buf = fd.read(dim * bytes_per_sample)
        size += dim * bytes_per_sample
        array = np.frombuffer(buf, dtype=np.dtype(dtype))

        if "M" in Type:  # As matrix
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
        b = fd.read(1)
        try:
            char = b.decode(encoding=default_encoding)
        except UnicodeDecodeError:
            raise ValueError("File format is wrong?")
        size += 1
        if char == " " or char == "\n":
            continue
        elif char == "[":
            hasparent = True
            break
        else:
            string.append(char)
            hasparent = False
            break

    # Read data
    ndmin = 1
    while True:
        char = fd.read(1).decode(encoding=default_encoding)
        size += 1
        if hasparent:
            if char == "]":
                char = fd.read(1).decode(encoding=default_encoding)
                size += 1
                assert char == "\n" or char == ""
                break
            elif char == "\n":
                ndmin = 2
            elif char == "":
                raise ValueError("There are no corresponding bracket ']' with '['")
        else:
            if char == "\n" or char == "":
                break
        string.append(char)
    string = "".join(string)
    assert len(string) != 0

    # Examine dtype
    match = re.match(r" *([^ \n]+) *", string)
    if match is None:
        dtype = np.float32
    else:
        ma = match.group(0)
        # If first element is integer, deal as interger array
        try:
            float(ma)
        except ValueError:
            raise RuntimeError(ma + "is not a digit\nFile format is wrong?")
        if "." in ma:
            dtype = np.float32
        else:
            dtype = np.int32
    array = np.loadtxt(StringIO(string), dtype=dtype, ndmin=ndmin)
    if return_size:
        return array, size
    else:
        return array


def _read_length_header(fd):
    (bytes_length,) = struct.unpack("<B", fd.read(1))
    length_ = from_bytes(fd.read(bytes_length))
    return length_


def _write_length_header(fd, length_):
    bit_length = length_.bit_length()
    bytes_length = int(math.ceil(bit_length / 8))
    fd.write(struct.pack("<B", bytes_length))
    fd.write(to_bytes(length_, bytes_length))
    return 1 + bytes_length


def save_ark(
    ark,
    array_dict,
    scp=None,
    append=False,
    text=False,
    endian="<",
    compression_method=None,
    write_function=None,
    write_kwargs=None,
):
    """Write ark

    Args:
        ark (str or fd):
        array_dict (dict):
        scp (str or fd):
        append (bool): If True is specified, open the file
            with appendable mode
        text (bool): If True, saving in text ark format.
        endian (str):
        compression_method (int):
        write_function: (str):
        write_kwargs: (Optional[dict]):
    """
    if write_kwargs is None:
        write_kwargs = {}

    if isinstance(ark, string_types):
        seekable = True
    # Maybe, never match with this
    elif not hasattr(ark, "tell"):
        seekable = False
    else:
        try:
            ark.tell()
            seekable = True
        except Exception:
            seekable = False

    if scp is not None and not isinstance(ark, string_types):
        if not seekable:
            raise TypeError(
                "scp file can be created only "
                "if the output ark file is a file or "
                "a seekable file descriptor."
            )

    # Write ark
    mode = "ab" if append else "wb"
    pos_list = []
    with open_or_fd(ark, mode) as fd:
        if seekable:
            offset = fd.tell()
        else:
            offset = 0
        size = 0
        for key in array_dict:
            encode_key = (key + " ").encode(encoding=default_encoding)
            fd.write(encode_key)
            size += len(encode_key)
            pos_list.append(size)
            data = array_dict[key]

            if write_function is not None:
                # Ignore case
                write_function = write_function.lower()

                if write_function == "soundfile":
                    import soundfile

                    def _write_function(fd, data):
                        if not isinstance(data, (list, tuple)):
                            raise TypeError(
                                "Expected list or tuple type, but got {}".format(
                                    type(data)
                                )
                            )
                        elif len(data) != 2:
                            raise ValueError(
                                "Expected length=2, bot got {}".format(len(data))
                            )
                        _fd = BytesIO()

                        if isinstance(data[0], np.ndarray) and isinstance(data[1], int):
                            _array, _rate = data

                        elif isinstance(data[1], np.ndarray) and isinstance(
                            data[0], int
                        ):
                            _rate, _array = data
                        else:
                            raise ValueError(
                                "Expected Tuple[int, np.ndarray] or "
                                "Tuple[np.ndarray, int]: "
                                "but got Tuple[{}, {}]".format(
                                    type(data[0]), type(data[1])
                                )
                            )

                        # By default, save as wav
                        if "format" not in write_kwargs:
                            write_kwargs["format"] = "wav"

                        soundfile.write(_fd, _array, _rate, **write_kwargs)
                        fd.write(b"AUDIO")
                        buf = _fd.getvalue()
                        # Write the information for the length
                        bytes_length = _write_length_header(fd, len(buf))
                        fd.write(buf)
                        return len(buf) + len(b"AUDIO") + bytes_length

                elif write_function == "pickle":

                    def _write_function(fd, data):
                        # Note that we don't need size information for pickle!

                        fd.write(b"PKL")
                        _fd = BytesIO()
                        pickle.dump(data, _fd, **write_kwargs)
                        buf = _fd.getbuffer()
                        fd.write(buf)
                        return len(buf) + len("PKL")

                elif write_function == "numpy":

                    def _write_function(fd, data):
                        # Write numpy file in BytesIO
                        _fd = BytesIO()
                        np.save(_fd, data, **write_kwargs)

                        fd.write(b"NPY")
                        buf = _fd.getvalue()

                        # Write the information for the length
                        bytes_length = _write_length_header(fd, len(buf))

                        # Write numpy to real file object
                        fd.write(buf)

                        return len(buf) + len(b"NPY") + bytes_length

                else:
                    raise RuntimeError(
                        "Not supported: write_function={}".format(write_function)
                    )

                size += _write_function(fd, data)

            elif isinstance(data, (list, tuple)):
                rate, array = data
                size += write_wav(fd, rate, array)
            elif text:
                size += write_array_ascii(fd, data, endian)
            else:
                size += write_array(fd, data, endian, compression_method)

    # Write scp
    mode = "a" if append else "w"
    if scp is not None:
        name = ark if isinstance(ark, string_types) else ark.name
        with open_or_fd(scp, mode) as fd:
            for key, position in zip(array_dict, pos_list):
                fd.write(key + " " + name + ":" + str(position + offset) + "\n")


def save_mat(fname, array, endian="<", compression_method=None):
    with open_or_fd(fname, "wb") as fd:
        return write_array(fd, array, endian, compression_method)


def write_array(fd, array, endian="<", compression_method=None):
    """Write array

    Args:
        fd (file): binary mode
        array (np.ndarray):
        endian (str):
    Returns:
        size (int):
    """
    size = 0
    assert isinstance(array, np.ndarray), type(array)
    fd.write(b"\0B")
    size += 2
    if compression_method is not None:
        if array.ndim != 2:
            raise ValueError(
                "array must be matrix if compression_method is not None: {}".format(
                    array.ndim
                )
            )

        global_header = GlobalHeader.compute(array, compression_method, endian)
        size += global_header.write(fd)
        if global_header.type == "CM":
            per_col_header = PerColHeader.compute(array, global_header)
            size += per_col_header.write(fd, global_header)

            array = per_col_header.float_to_char(array.T)

            byte_string = array.tobytes()
            fd.write(byte_string)
            size += len(byte_string)

        elif global_header.type == "CM2":
            array = global_header.float_to_uint(array)

            byte_string = array.tobytes()
            fd.write(byte_string)
            size += len(byte_string)

        elif global_header.type == "CM3":
            array = global_header.float_to_uint(array)

            byte_string = array.tobytes()
            fd.write(byte_string)
            size += len(byte_string)

    elif array.dtype == np.int32:
        assert array.ndim == 1, array.ndim  # Must be vector
        fd.write(b"\4")
        fd.write(struct.pack(endian + "i", len(array)))
        for x in array:
            fd.write(b"\4")
            fd.write(struct.pack(endian + "i", x))
        size += (len(array) + 1) * 5

    elif array.dtype == np.float32 or array.dtype == np.float64:
        assert 0 < len(array.shape) < 3  # Matrix or vector
        if len(array.shape) == 1:
            if array.dtype == np.float32:
                fd.write(b"FV ")
                size += 3
            elif array.dtype == np.float64:
                fd.write(b"DV ")
                size += 3
            fd.write(b"\4")
            size += 1
            fd.write(struct.pack(endian + "i", len(array)))
            size += 4

        elif len(array.shape) == 2:
            if array.dtype == np.float32:
                fd.write(b"FM ")
                size += 3
            elif array.dtype == np.float64:
                fd.write(b"DM ")
                size += 3
            fd.write(b"\4")
            size += 1
            fd.write(struct.pack(endian + "i", len(array)))  # Rows
            size += 4

            fd.write(b"\4")
            size += 1
            fd.write(struct.pack(endian + "i", array.shape[1]))  # Cols
            size += 4
        if endian not in array.dtype.str:
            array = array.astype(array.dtype.newbyteorder())
        fd.write(array.tobytes())
        size += array.nbytes
    else:
        raise ValueError("Unsupported array type: {}".format(array.dtype))
    return size


def write_array_ascii(fd, array, digit=".12g"):
    """write_array_ascii

    Args:
        fd (file): binary mode
        array (np.ndarray):
        digit (str):
    Returns:
        size (int):
    """
    assert isinstance(array, np.ndarray), type(array)
    assert array.ndim in (1, 2), array.ndim
    size = 0
    fd.write(b" [")
    size += 2
    if array.ndim == 2:
        for row in array:
            fd.write(b"\n  ")
            size += 3
            for i in row:
                string = format(i, digit)
                fd.write(string.encode(encoding=default_encoding))
                fd.write(b" ")
                size += len(string) + 1
        fd.write(b"]\n")
        size += 2
    elif array.ndim == 1:
        fd.write(b" ")
        size += 1
        for i in array:
            string = format(i, digit)
            fd.write(string.encode(encoding=default_encoding))
            fd.write(b" ")
            size += len(string) + 1
        fd.write(b"]\n")
        size += 2
    return size
