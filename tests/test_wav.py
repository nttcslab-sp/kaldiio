import numpy as np

from kaldiio.matio import load_ark
from kaldiio.matio import load_scp
from kaldiio.matio import save_ark
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


def test_load_wav_stream(tmpdir):
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
