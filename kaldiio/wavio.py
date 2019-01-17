from __future__ import unicode_literals

from io import BytesIO
import wave

import numpy as np
from scipy.io import wavfile as wavfile

from kaldiio.utils import seekable


def read_wav(fd, return_size=False):
    wd = wave.open(fd)
    rate = wd.getframerate()
    nchannels = wd.getnchannels()
    nbytes = wd.getsampwidth()
    if nbytes == 1:
        # 8bit-PCM is unsigned
        dtype = 'uint8'
    elif nbytes == 2:
        dtype = 'int16'
    elif nbytes == 4:
        dtype = 'int32'
    elif nbytes == 8:
        dtype = 'int64'
    else:
        raise ValueError('bytes_per_sample must be 1, 2, 4 or 8')
    data = wd.readframes(wd.getnframes())
    size = 44 + len(data)
    array = np.frombuffer(data, dtype=np.dtype(dtype))
    if nchannels > 1:
        array = array.reshape(-1, nchannels)

    if return_size:
        return (rate, array), size
    else:
        return rate, array


def read_wav_scipy(fd, return_size=False):
    if not seekable(fd):
        # scipy.io.wavfile doesn't support unseekable fd
        data = fd.read()
        fd = BytesIO(data)
        offset = None
    else:
        offset = fd.tell()
    rate, array = wavfile.read(fd)
    size = 44 + array.nbytes
    if offset is not None:
        fd.seek(size + offset)

    if return_size:
        return (rate, array), size
    else:
        return rate, array


def write_wav(fd, rate, array):
    if array.dtype == np.int16:
        sampwidth = 2
    elif array.dtype == np.int32:
        sampwidth = 4
    elif array.dtype == np.float16:
        sampwidth = 2
    elif array.dtype == np.float16:
        sampwidth = 4
    else:
        raise ValueError('Not Supported dtype {}'.format(array.dtype))

    if array.ndim == 2:
        nchannels = array.shape[1]
    elif array.ndim == 1:
        nchannels = 1
    else:
        raise ValueError(
            'Not Supported dimension: 0 or 1, but got {}'.format(array.ndim))

    w = wave.Wave_write(fd)
    w.setnchannels(nchannels)
    w.setsampwidth(sampwidth)
    w.setframerate(rate)
    data = array.tobytes()
    w.writeframes(data)

    return 44 + len(data)
