import numpy as np

from kaldiio.matio import load_ark
from kaldiio.matio import load_scp
from kaldiio.matio import save_ark
from kaldiio.utils import open_like_kaldi
from kaldiio.wavio import write_wav


def test_load_wav(tmpdir):
    path = tmpdir.mkdir('test')
    wav = path.join('a.wav').strpath
    scp = path.join('wav.scp').strpath

    # Write as pcm16
    array = np.random.randint(0, 10, 10, dtype=np.int16)
    write_wav(wav, 8000, array)
    with open(scp, 'w') as f:
        f.write('aaa {wav}\n'.format(wav=wav))
    rate, array2 = list(load_scp(scp).values())[0]
    np.testing.assert_array_equal(array, array2)


def test_read_write_wav(tmpdir):
    path = tmpdir.mkdir('test')
    ark = path.join('a.ark').strpath
    scp = path.join('a.scp').strpath

    # Write as pcm16
    array = np.random.randint(0, 10, 10, dtype=np.int16)
    array2 = np.random.randint(0, 10, 10, dtype=np.int16)
    d = {'utt': (8000, array), 'utt2': (8000, array2)}
    save_ark(ark, d, scp=scp)

    d = load_scp(scp)
    rate, test = d['utt']
    assert rate == 8000
    np.testing.assert_array_equal(array, test)

    rate, test = d['utt2']
    assert rate == 8000
    np.testing.assert_array_equal(array2, test)

    d = dict(load_ark(ark))
    rate, test = d['utt']
    assert rate == 8000
    np.testing.assert_array_equal(array, test)

    rate, test = d['utt2']
    assert rate == 8000
    np.testing.assert_array_equal(array2, test)


def test_scpwav_stream(tmpdir):
    path = tmpdir.mkdir('test')
    wav = path.join('aaa.wav').strpath
    wav2 = path.join('bbb.wav').strpath
    scp = path.join('wav.scp').strpath

    # Write as pcm16
    array = np.random.randint(0, 10, 10, dtype=np.int16)
    write_wav(wav, 8000, array)

    array2 = np.random.randint(0, 10, 10, dtype=np.int16)
    write_wav(wav2, 8000, array2)

    with open(scp, 'w') as f:
        f.write('aaa cat {wav} |\n'.format(wav=wav))
        f.write('bbb cat {wav} |\n'.format(wav=wav2))
    rate, test = load_scp(scp)['aaa']
    rate, test2 = load_scp(scp)['bbb']
    np.testing.assert_array_equal(array, test)
    np.testing.assert_array_equal(array2, test2)


def test_wavark_stream(tmpdir):
    path = tmpdir.mkdir('test')
    ark = path.join('a.ark').strpath

    # Write as pcm16
    array = np.random.randint(0, 10, 10, dtype=np.int16)
    array2 = np.random.randint(0, 10, 10, dtype=np.int16)
    d = {'utt': (8000, array), 'utt2': (8000, array2)}
    save_ark(ark, d)

    with open_like_kaldi('cat {}|'.format(ark), 'rb') as f:
        d = dict(load_ark(f))
        rate, test = d['utt']
        assert rate == 8000
        np.testing.assert_array_equal(array, test)

        rate, test = d['utt2']
        assert rate == 8000
        np.testing.assert_array_equal(array2, test)


def test_segments(tmpdir):
    # Create wav.scp
    path = tmpdir.mkdir('test')
    wavscp = path.join('wav.scp').strpath

    rate = 500
    with open(wavscp, 'w') as f:
        wav = path.join('0.wav').strpath
        array0 = np.random.randint(0, 10, 2000, dtype=np.int16)
        write_wav(wav, rate, array0)
        f.write('wav0 {}\n'.format(wav))

        wav = path.join('1.wav').strpath
        array1 = np.random.randint(0, 10, 2000, dtype=np.int16)
        write_wav(wav, rate, array1)
        f.write('wav1 {}\n'.format(wav))

    # Create segments
    segments = path.join('segments').strpath
    with open(segments, 'w') as f:
        f.write('utt1 wav0 0.1 0.2\n')
        f.write('utt2 wav0 0.4 0.6\n')
        f.write('utt3 wav1 0.4 0.5\n')
        f.write('utt4 wav1 0.6 0.8\n')
    d = load_scp(wavscp, segments=segments)

    np.testing.assert_array_equal(
        d['utt1'][1], array0[int(0.1 * rate):int(0.2 * rate)])
    np.testing.assert_array_equal(
        d['utt2'][1], array0[int(0.4 * rate):int(0.6 * rate)])
    np.testing.assert_array_equal(
        d['utt3'][1], array1[int(0.4 * rate):int(0.5 * rate)])
    np.testing.assert_array_equal(
        d['utt4'][1], array1[int(0.6 * rate):int(0.8 * rate)])
