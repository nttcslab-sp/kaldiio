import numpy as np
import wave

import kaldiio.python_wave as wave

def read_wav(fd, return_size=False):
    wd = wave.open(fd)
    rate = wd.getframerate()
    nchannels = wd.getnchannels()
    nbytes = wd.getsampwidth()
    if nbytes == 1:
        # 8bit-PCM is unsigned
        dtype = "uint8"
    elif nbytes == 2:
        dtype = "int16"
    else:
        raise ValueError("bytes_per_sample must be 1, 2, 4 or 8")
    data = wd.readframes(wd.getnframes())
    size = 44 + len(data)
    array = np.frombuffer(data, dtype=np.dtype(dtype))
    if nchannels > 1:
        array = array.reshape(-1, nchannels)

    if return_size:
        return (rate, array), size
    else:
        return rate, array


def write_wav(fd, rate, array):
    if array.dtype == np.uint8:
        sampwidth = 1
    elif array.dtype == np.int16:
        sampwidth = 2
    else:
        raise ValueError("Not Supported dtype {}".format(array.dtype))

    if array.ndim == 2:
        nchannels = array.shape[1]
    elif array.ndim == 1:
        nchannels = 1
    else:
        raise ValueError(
            "Not Supported dimension: 0 or 1, but got {}".format(array.ndim)
        )

    w = wave.Wave_write(fd)
    w.setnchannels(nchannels)
    w.setsampwidth(sampwidth)
    w.setframerate(rate)
    data = array.tobytes()
    w.writeframes(data)

    return 44 + len(data)
