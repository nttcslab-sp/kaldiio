from io import BytesIO
import wave

import numpy as np
from scipy.io import wavfile as wavfile


def read_wav(fd, offset=None, return_size=False):
    if offset is not None:
        fd.seek(offset)
    wd = wave.open(fd)
    assert isinstance(wd, wave.Wave_read)
    rate = wd.getframerate()
    nchannels = wd.getnchannels()
    nbytes = wd.getsampwidth()
    if nbytes == 2:
        dtype = 'int16'
    elif nbytes == 4:
        dtype = 'int32'
    elif nbytes == 8:
        dtype = 'int64'
    else:
        raise ValueError('bytes_per_sample must be 2, 4 or 8')
    data = wd.readframes(wd.getnframes())
    size = 44 + len(data)
    array = np.frombuffer(data, dtype=np.dtype(dtype))
    if nchannels > 1:
        array = array.reshape(-1, nchannels)

    if return_size:
        return (rate, array), size
    else:
        return rate, array


def read_wav_scipy(fd, offset=None, return_size=False):
    if offset is not None:
        fd.seek(offset)
    if not fd.seekable():
        # scipy.io.wavfile doesn't support unseekable fd
        data = fd.read()
        fd = BytesIO(data)
    rate, array = wavfile.read(fd)
    size = 44 + array.nbytes

    if return_size:
        return (rate, array), size
    else:
        return rate, array
