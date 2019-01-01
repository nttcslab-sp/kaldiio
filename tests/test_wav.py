import wave

import numpy as np

from kaldiio.matio import load_scp


def test_load_wav(tmpdir):
    path = tmpdir.mkdir('test')
    wav = path.join('a.wav').strpath
    scp = path.join('wav.scp').strpath

    # Write as pcm16
    array = np.random.randn(3, 4).astype(np.int16)
    _wavwrite(wav, array, 8000)
    with open(scp, 'w') as f:
        f.write('aaa {wav}'.format(wav=wav))
    rate, array2 = list(load_scp(scp).values())[0]
    np.testing.assert_array_equal(array, array2)


def test_load_wav_stream(tmpdir):
    path = tmpdir.mkdir('test')
    wav = path.join('a.wav').strpath
    scp = path.join('wav.scp').strpath

    # Write as pcm16
    array = np.random.randn(3, 4).astype(np.int16)
    _wavwrite(wav, array, 8000)
    with open(scp, 'w') as f:
        f.write('aaa cat {wav} |'.format(wav=wav))
    rate, array2 = list(load_scp(scp).values())[0]
    np.testing.assert_array_equal(array, array2)


def _wavwrite(fd, array, framerate):
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
    w.setframerate(framerate)
    w.writeframes(array.tobytes())
    w.close()
