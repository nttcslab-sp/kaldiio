import os

import numpy as np
import pytest

from kaldiio.matio import load_ark
from kaldiio.matio import load_scp
from kaldiio.matio import load_scp_sequential
from kaldiio.matio import save_ark
from kaldiio.utils import open_like_kaldi


@pytest.mark.parametrize('dtype', [np.int16])
@pytest.mark.parametrize('func', [load_scp, load_scp_sequential])
@pytest.mark.parametrize('write_function', ["soundfile", "pickle", "numpy"])
def test_read_write(tmpdir, func, dtype, write_function):
    path = tmpdir.mkdir('test')
    ark = path.join('a.ark').strpath
    scp = path.join('a.scp').strpath

    # Write as pcm16
    array = np.random.randint(-1000, 1000, 100).astype(np.double) / abs(np.iinfo(np.int16).min)
    array2 = np.random.randint(-1000, 1000, 100) / abs(np.iinfo(np.int16).min)

    if write_function == "numpy":
        d = {'utt': array, 'utt2': array2}
    else:
        d = {'utt': (8000, array), 'utt2': (8000, array2)}
    save_ark(ark, d, scp=scp, write_function=write_function)

    d = dict(func(scp))
    if write_function == "numpy":
        test = d['utt']
    else:
        rate, test = d['utt']
        assert rate == 8000
    np.testing.assert_allclose(array, test)

    if write_function == "numpy":
        test = d['utt2']
    else:
        rate, test = d['utt2']
        assert rate == 8000
    np.testing.assert_allclose(array2, test)

    d = dict(load_ark(ark))
    if write_function == "numpy":
        test = d['utt']
    else:
        rate, test = d['utt']
        assert rate == 8000
    np.testing.assert_allclose(array, test)

    if write_function == "numpy":
        test = d['utt2']
    else:
        rate, test = d['utt2']
        assert rate == 8000
    np.testing.assert_allclose(array2, test)


@pytest.mark.parametrize('dtype', [np.int16])
@pytest.mark.parametrize('write_function', ["soundfile", "pickle", "numpy"])
def test_wavark_stream(tmpdir, dtype, write_function):
    path = tmpdir.mkdir('test')
    ark = path.join('a.ark').strpath

    # Write as pcm16
    array = np.random.randint(-1000, 1000, 100).astype(np.double) / abs(np.iinfo(np.int16).min)
    array2 = np.random.randint(-1000, 1000, 100) / abs(np.iinfo(np.int16).min)
    if write_function == "numpy":
        d = {'utt': array, 'utt2': array2}
    else:
        d = {'utt': (8000, array), 'utt2': (8000, array2)}
    save_ark(ark, d, write_function=write_function)

    with open_like_kaldi('cat {}|'.format(ark), 'rb') as f:
        d = dict(load_ark(f))
        if write_function == "numpy":
            test = d['utt']
        else:
            rate, test = d['utt']
            assert rate == 8000
        np.testing.assert_allclose(array, test)

        if write_function == "numpy":
            test = d['utt2']
        else:
            rate, test = d['utt2']
            assert rate == 8000
        np.testing.assert_allclose(array2, test)
