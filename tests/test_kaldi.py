# -*- coding: utf-8 -*-
import glob
import os
import sys
import wave

import numpy as np
import pytest

import kaldiio


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


def test_read_arks():
    # Assume arks dir existing at the same directory
    arkdir = os.path.join(os.path.dirname(__file__), 'arks')
    arks = {fname: dict(kaldiio.load_ark(fname))
            for fname in glob.glob(os.path.join(arkdir, '*.ark'))}
    fnames = list(arks)
    for fname in fnames[1:]:
        _compare_allclose(arks[fname], arks[fnames[0]], atol=1e-1)


def test_write_load(tmpdir):
    path = tmpdir.mkdir('test')

    a = np.random.rand(1000, 120).astype(np.float32)
    b = np.random.rand(10, 120).astype(np.float32)
    origin = {'a': a, 'b': b}
    kaldiio.save_ark(path.join('a.ark').strpath, origin,
                     scp=path.join('b.scp').strpath)

    d2 = {k: v for k, v in
          kaldiio.load_ark(path.join('a.ark').strpath)}
    d5 = {k: v for k, v in
          kaldiio.load_scp(path.join('b.scp').strpath).items()}
    with open(path.join('a.ark').strpath, 'rb') as fd:
        d6 = {k: v for k, v in
              kaldiio.load_ark(fd)}
    _compare(d2, origin)
    _compare(d5, origin)
    _compare(d6, origin)


def test_write_load_ascii(tmpdir):
    path = tmpdir.mkdir('test')
    a = np.random.rand(10, 10).astype(np.float32)
    b = np.random.rand(5, 35).astype(np.float32)
    origin = {'a': a, 'b': b}
    kaldiio.save_ark(path.join('a.ark').strpath, origin,
                     scp=path.join('a.scp').strpath, text=True)
    d2 = {k: v for k, v in
          kaldiio.load_ark(path.join('a.ark').strpath)}
    d5 = {k: v for k, v in
          kaldiio.load_scp(path.join('a.scp').strpath).items()}
    _compare_allclose(d2, origin)
    _compare_allclose(d5, origin)


def test_append_mode(tmpdir):
    path = tmpdir.mkdir('test')

    a = np.random.rand(1000, 120).astype(np.float32)
    b = np.random.rand(10, 120).astype(np.float32)
    origin = {'a': a, 'b': b}
    kaldiio.save_ark(path.join('a.ark').strpath, origin,
                     scp=path.join('b.scp').strpath)

    kaldiio.save_ark(path.join('a2.ark').strpath, {'a': a},
                     scp=path.join('b2.scp').strpath, append=True)
    kaldiio.save_ark(path.join('a2.ark').strpath, {'b': b},
                     scp=path.join('b2.scp').strpath, append=True)
    d1 = {k: v for k, v in
          kaldiio.load_ark(path.join('a.ark').strpath)}
    d2 = {k: v for k, v in
          kaldiio.load_scp(path.join('b.scp').strpath).items()}
    d3 = {k: v for k, v in
          kaldiio.load_ark(path.join('a2.ark').strpath)}
    d4 = {k: v for k, v in
          kaldiio.load_scp(path.join('b2.scp').strpath).items()}
    _compare(d1, origin)
    _compare(d2, origin)
    _compare(d3, origin)
    _compare(d4, origin)


def test_load_wav(tmpdir):
    path = tmpdir.mkdir('test')
    wav = path.join('a.wav').strpath
    scp = path.join('wav.scp').strpath

    # Write as pcm16
    array = np.random.randn(3, 4).astype(np.int16)
    _wavwrite(wav, array, 8000)
    with open(scp, 'w') as f:
        f.write('aaa cat {wav} |'.format(wav=wav))
    with open(scp, 'w') as f:
        f.write('aaa {wav}'.format(wav=wav))
    rate, array2 = list(kaldiio.load_wav_scp(scp).values())[0]
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
    rate, array2 = list(kaldiio.load_wav_scp(scp).values())[0]
    np.testing.assert_array_equal(array, array2)


def _compare(d1, d2):
    assert len(d1) != 0
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        np.testing.assert_array_equal(d1[key], d2[key])


def _compare_allclose(d1, d2, rtol=1e-07, atol=0.):
    assert len(d1) != 0
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        np.testing.assert_allclose(d1[key], d2[key], rtol, atol)


def test_open_like_kaldi(tmpdir):
    with kaldiio.open_like_kaldi('echo -n hello |', 'r') as f:
        assert f.read() == 'hello'
    txt = tmpdir.mkdir('test').join('out.txt').strpath
    with kaldiio.open_like_kaldi('| cat > {}'.format(txt), 'w') as f:
        f.write('hello')
    with open(txt, 'r') as f:
        assert f.read() == 'hello'


def test_open_stdsteadm(tmpdir):
    with kaldiio.open_like_kaldi('-', 'w') as f:
        assert f is sys.stdout
    with kaldiio.open_like_kaldi('-', 'wb'):
        pass
    with kaldiio.open_like_kaldi('-', 'r') as f:
        assert f is sys.stdin
    with kaldiio.open_like_kaldi('-', 'rb'):
        pass


if __name__ == '__main__':
    pytest.main()
