import os

import numpy as np
import pytest

from kaldiio.matio import load_ark
from kaldiio.matio import load_scp
from kaldiio.matio import load_scp_sequential
from kaldiio.matio import save_ark
from kaldiio.utils import open_like_kaldi
from kaldiio.wavio import read_wav
from kaldiio.wavio import write_wav


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
@pytest.mark.parametrize("func", [read_wav])
def test_read_wav(tmpdir, func, dtype):
    path = tmpdir.mkdir("test")
    wav = path.join("a.wav").strpath
    # Write as pcm16
    array = np.random.randint(0, 10, 10, dtype=dtype)
    write_wav(wav, 8000, array)
    with open(wav, "rb") as f:
        rate, array2 = func(f)
    np.testing.assert_array_equal(array, array2)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
@pytest.mark.parametrize("func", [load_scp, load_scp_sequential])
def test_load_wav_scp(tmpdir, func, dtype):
    path = tmpdir.mkdir("test")
    wav = path.join("a.wav").strpath
    scp = path.join("wav.scp").strpath

    # Write as pcm16
    array = np.random.randint(0, 10, 10, dtype=dtype)
    write_wav(wav, 8000, array)
    with open(scp, "w") as f:
        f.write("aaa {wav}\n".format(wav=wav))
    rate, array2 = list(dict(func(scp)).values())[0]
    np.testing.assert_array_equal(array, array2)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
@pytest.mark.parametrize("func", [load_scp, load_scp_sequential])
def test_read_write_wav(tmpdir, func, dtype):
    path = tmpdir.mkdir("test")
    ark = path.join("a.ark").strpath
    scp = path.join("a.scp").strpath

    # Write as pcm16
    array = np.random.randint(0, 10, 10, dtype=dtype)
    array2 = np.random.randint(0, 10, 10, dtype=dtype)
    d = {"utt": (8000, array), "utt2": (8000, array2)}
    save_ark(ark, d, scp=scp)

    d = dict(func(scp))
    rate, test = d["utt"]
    assert rate == 8000
    np.testing.assert_array_equal(array, test)

    rate, test = d["utt2"]
    assert rate == 8000
    np.testing.assert_array_equal(array2, test)

    d = dict(load_ark(ark))
    rate, test = d["utt"]
    assert rate == 8000
    np.testing.assert_array_equal(array, test)

    rate, test = d["utt2"]
    assert rate == 8000
    np.testing.assert_array_equal(array2, test)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
@pytest.mark.parametrize("func", [load_scp, load_scp_sequential])
def test_scpwav_stream(tmpdir, func, dtype):
    path = tmpdir.mkdir("test")
    wav = path.join("aaa.wav").strpath
    wav2 = path.join("bbb.wav").strpath
    scp = path.join("wav.scp").strpath

    # Write as pcm16
    array = np.random.randint(0, 10, 40, dtype=dtype).reshape(5, 8)
    write_wav(wav, 8000, array)

    array2 = np.random.randint(0, 10, 10, dtype=dtype)
    write_wav(wav2, 8000, array2)

    with open(scp, "w") as f:
        f.write("aaa sox {wav} -t wav - |\n".format(wav=wav))
        f.write("bbb cat {wav} |\n".format(wav=wav2))
    rate, test = dict(func(scp))["aaa"]
    rate, test2 = dict(func(scp))["bbb"]
    np.testing.assert_array_equal(array, test)
    np.testing.assert_array_equal(array2, test2)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
def test_wavark_stream(tmpdir, dtype):
    path = tmpdir.mkdir("test")
    ark = path.join("a.ark").strpath

    # Write as pcm16
    array = np.random.randint(0, 10, 10, dtype=dtype)
    array2 = np.random.randint(0, 10, 10, dtype=dtype)
    d = {"utt": (8000, array), "utt2": (8000, array2)}
    save_ark(ark, d)

    with open_like_kaldi("cat {}|".format(ark), "rb") as f:
        d = dict(load_ark(f))
        rate, test = d["utt"]
        assert rate == 8000
        np.testing.assert_array_equal(array, test)

        rate, test = d["utt2"]
        assert rate == 8000
        np.testing.assert_array_equal(array2, test)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
@pytest.mark.parametrize("func", [load_scp, load_scp_sequential])
def test_segments(tmpdir, func, dtype):
    # Create wav.scp
    path = tmpdir.mkdir("test")
    wavscp = path.join("wav.scp").strpath

    rate = 500
    with open(wavscp, "w") as f:
        wav = path.join("0.wav").strpath
        array0 = np.random.randint(0, 10, 2000, dtype=dtype)
        write_wav(wav, rate, array0)
        f.write("wav0 {}\n".format(wav))

        wav = path.join("1.wav").strpath
        array1 = np.random.randint(0, 10, 2000, dtype=dtype)
        write_wav(wav, rate, array1)
        f.write("wav1 {}\n".format(wav))

    # Create segments
    segments = path.join("segments").strpath
    with open(segments, "w") as f:
        f.write("utt1 wav0 0.1 0.2\n")
        f.write("utt2 wav0 0.4 0.6\n")
        f.write("utt3 wav1 0.4 0.5\n")
        f.write("utt4 wav1 0.6 0.8\n")
    d = dict(func(wavscp, segments=segments))

    np.testing.assert_array_equal(
        d["utt1"][1], array0[int(0.1 * rate) : int(0.2 * rate)]
    )
    np.testing.assert_array_equal(
        d["utt2"][1], array0[int(0.4 * rate) : int(0.6 * rate)]
    )
    np.testing.assert_array_equal(
        d["utt3"][1], array1[int(0.4 * rate) : int(0.5 * rate)]
    )
    np.testing.assert_array_equal(
        d["utt4"][1], array1[int(0.6 * rate) : int(0.8 * rate)]
    )


@pytest.mark.parametrize("func", [load_scp, load_scp_sequential])
def test_incorrect_header_wav(tmpdir, func):
    wav = os.path.join(os.path.dirname(__file__), "arks", "incorrect_header.wav")
    _, array = read_wav(wav)
    path = tmpdir.mkdir("test")
    scp = path.join("wav.scp").strpath

    with open(scp, "w") as f:
        f.write("aaa sox {wav} -t wav - |\n".format(wav=wav))
    rate, test = dict(func(scp))["aaa"]
    np.testing.assert_array_equal(array, test)
