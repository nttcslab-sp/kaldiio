import struct

import numpy as np


kAutomaticMethod = 1
kSpeechFeature = 2
kTwoByteAuto = 3
kTwoByteSignedInteger = 4
kOneByteAuto = 5
kOneByteUnsignedInteger = 6
kOneByteZeroOne = 7


class GlobalHeader(object):
    """This is a imitation class of the structure "GlobalHeader" """
    def __init__(self, type, min_value, range, rows, cols):
        if type in ('CM', 'CM2'):
            c = 65535.
        elif type == 'CM3':
            c = 255.
        else:
            raise RuntimeError('Not supported type={}'.format(type))
        self.type = type
        self.c = c
        self.min_value = min_value
        self.range = range
        self.rows = rows
        self.cols = cols

    @property
    def size(self):
        return 17 + len(self.type)

    @staticmethod
    def read(fd, type='CM', endian='<'):
        min_value = struct.unpack(endian + 'f', fd.read(4))[0]
        range = struct.unpack(endian + 'f', fd.read(4))[0]
        rows = struct.unpack(endian + 'i', fd.read(4))[0]
        cols = struct.unpack(endian + 'i', fd.read(4))[0]
        assert rows > 0
        assert cols > 0
        return GlobalHeader(type, min_value, range, rows, cols)

    def write(self, fd, endian='<'):
        fd.write(self.type.encode() + b' ')
        fd.write(struct.pack(endian + 'f', self.min_value))
        fd.write(struct.pack(endian + 'f', self.range))
        fd.write(struct.pack(endian + 'i', self.rows))
        fd.write(struct.pack(endian + 'i', self.cols))
        return self.size

    @staticmethod
    def compute(array, compression_method):
        if compression_method == kAutomaticMethod:
            if array.shape[0] > 8:
                compression_method = kSpeechFeature
            else:
                compression_method = kTwoByteAuto

        if compression_method == kSpeechFeature:
            matrix_type = 'CM'
        elif compression_method == kTwoByteAuto or \
                compression_method == kTwoByteSignedInteger:
            matrix_type = 'CM2'
        elif compression_method == kOneByteAuto or \
                compression_method == kOneByteUnsignedInteger or \
                compression_method == kOneByteZeroOne:
            matrix_type = 'CM3'
        else:
            raise ValueError(
                'Unknown compression_method: {}'.format(compression_method))

        if compression_method == kSpeechFeature or \
                compression_method == kTwoByteAuto or \
                compression_method == kOneByteAuto:
            min_value = array.min()
            max_value = array.max()
            if min_value == max_value:
                max_value = min_value + (1. + abs(min_value))
            range_ = max_value - min_value
        elif compression_method == kTwoByteSignedInteger:
            min_value = -32768.
            range_ = 65535.
        elif compression_method == kOneByteUnsignedInteger:
            min_value = 0.
            range_ = 255.
        elif compression_method == kOneByteZeroOne:
            min_value = 0.
            range_ = 1.
        else:
            raise ValueError(
                'Unknown compression_method: {}'.format(compression_method))

        return GlobalHeader(
            matrix_type, min_value, range_, array.shape[0], array.shape[1])

    def float_to_uint(self, array):
        return (array - self.min_value) / self.range * self.c

    def uint_to_float(self, array):
        return self.min_value + array * self.range / self.c


class PerColHeader(object):
    """This is a imitation class of the structure "PerColHeader" """
    def __init__(self, p0, p25, p75, p100):
        # p means percentile
        self.p0 = p0
        self.p25 = p25
        self.p75 = p75
        self.p100 = p100

    @property
    def size(self):
        return 8 * self.p0.shape[0]

    @staticmethod
    def read(fd, global_header, endian='<'):
        # Read PerColHeader
        size_of_percolheader = 8
        buf = fd.read(size_of_percolheader * global_header.cols)
        header_array = np.frombuffer(buf, dtype=np.dtype(endian + 'u2'))
        header_array = np.asarray(header_array, np.float32)
        # Decompress header
        header_array = global_header.uint_to_float(header_array)
        header_array = header_array.reshape(-1, 4, 1)
        return PerColHeader(header_array[:, 0], header_array[:, 1],
                            header_array[:, 2], header_array[:, 3])

    def write(self, fd, global_header, endian='<'):
        header_array = np.concatenate(
            [self.p0, self.p25, self.p75, self.p100], axis=1)
        header_array = global_header.float_to_uint(header_array)
        header_array = header_array.astype(np.dtype(endian + 'u2'))
        byte_str = header_array.tobytes()
        fd.write(byte_str)
        return len(byte_str)

    @staticmethod
    def compute(array, global_header):
        quarter_nr = array.shape[0] // 4
        if array.shape[0] >= 5:
            srows = np.partition(
                array,
                [0, quarter_nr, 3 * quarter_nr, array.shape[0] - 1], axis=0)
            p0 = srows[0]
            p25 = srows[quarter_nr]
            p75 = srows[3 * quarter_nr]
            p100 = srows[array.shape[0] - 1]
        else:
            srows = np.sort(array, axis=0)
            p0 = srows[0]
            if array.shape[0] > 1:
                p25 = srows[1]
            else:
                p25 = p0 + 1
            if array.shape[0] > 2:
                p75 = srows[2]
            else:
                p75 = p25 + 1
            if array.shape[0] > 3:
                p100 = srows[3]
            else:
                p100 = p75 + 1
        p0 = global_header.float_to_uint(p0)
        p25 = global_header.float_to_uint(p25)
        p75 = global_header.float_to_uint(p75)
        p100 = global_header.float_to_uint(p100)

        p0 = np.minimum(p0, 65532)
        p25 = np.minimum(np.maximum(p25, p0 + 1), 65533)
        p75 = np.minimum(np.maximum(p75, p25 + 1), 65534)
        p100 = np.maximum(p100, p75 + 1)

        p0 = global_header.uint_to_float(p0)
        p25 = global_header.uint_to_float(p25)
        p75 = global_header.uint_to_float(p75)
        p100 = global_header.uint_to_float(p100)

        p0 = p0[:, None]
        p25 = p25[:, None]
        p75 = p75[:, None]
        p100 = p100[:, None]
        return PerColHeader(p0, p25, p75, p100)

    def float_to_char(self, array):
        p0, p25, p75, p100 = self.p0, self.p25, self.p75, self.p100

        ma1 = array < p25
        ma3 = array >= p75
        ma2 = ~ma1 * ~ma3

        tmp = (array - p0) / (p25 - p0) * 64. + 0.5
        tmp = np.where(tmp < 0., 0., np.where(tmp > 64., 64., tmp))

        tmp2 = ((array - p25) / (p75 - p25) * 128. + 64.5)
        tmp2 = np.where(tmp2 < 64., 64., np.where(tmp2 > 192., 192., tmp2))

        tmp3 = ((array - p75) / (p100 - p75) * 63. + 192.5)
        tmp3 = np.where(tmp3 < 192., 192., np.where(tmp3 > 255., 255., tmp3))
        array = np.where(ma1, tmp, np.where(ma2, tmp2, tmp3))
        return array

    def char_to_float(self, array):
        p0, p25, p75, p100 = self.p0, self.p25, self.p75, self.p100

        ma1 = array <= 64
        ma3 = array > 192
        ma2 = ~ma1 * ~ma3  # 192 >= array > 64

        return np.where(
            ma1, p0 + (p25 - p0) * array * (1 / 64.),
            np.where(ma2, p25 + (p75 - p25) * (array - 64.) * (1 / 128.),
                     p75 + (p100 - p75) * (array - 192.) * (1 / 63.)))
