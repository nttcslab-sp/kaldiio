import numpy

from kaldiio.highlevel import ReadHelper
from kaldiio.highlevel import WriteHelper
from kaldiio.matio import load_ark
from kaldiio.matio import load_scp
from kaldiio.matio import save_ark
from kaldiio.wavio import write_wav


def test_read_helper(tmpdir):
    path = tmpdir.strpath
    array_in = numpy.random.randn(10, 10)
    save_ark('{}/feats.ark'.format(path),
             {'foo': array_in}, scp='{}/feats.scp'.format(path))
    helper = ReadHelper('ark:cat {}/feats.ark |'.format(path))
    for uttid, array_out in helper:
        assert uttid == 'foo'
        numpy.testing.assert_array_equal(array_in, array_out)

    helper = ReadHelper('scp:{}/feats.scp'.format(path))
    for uttid, array_out in helper:
        assert uttid == 'foo'
        numpy.testing.assert_array_equal(array_in, array_out)


def test_read_helper_ascii(tmpdir):
    path = tmpdir.strpath
    array_in = numpy.random.randn(10, 10)
    save_ark('{}/feats.ark'.format(path),
             {'foo': array_in}, scp='{}/feats.scp'.format(path),
             text=True)
    helper = ReadHelper('ark:cat {}/feats.ark |'.format(path))
    for uttid, array_out in helper:
        assert uttid == 'foo'
        numpy.testing.assert_allclose(array_in, array_out)

    helper = ReadHelper('ark:{}/feats.ark'.format(path))
    for uttid, array_out in helper:
        assert uttid == 'foo'
        numpy.testing.assert_allclose(array_in, array_out)


def test_write_helper(tmpdir):
    path = tmpdir.strpath
    d = {'foo': numpy.random.randn(10, 10),
         'bar': numpy.random.randn(10, 10)}

    with WriteHelper('ark,f,scp:{p}/out.ark,{p}/out.scp'.format(p=path)) as w:
        for k, v in d.items():
            w(k, v)
    from_ark = dict(load_ark('{p}/out.ark'.format(p=path)))
    from_scp = load_scp('{p}/out.scp'.format(p=path))
    _compare(from_ark, d)
    _compare(from_scp, d)


def test_write_helper_ascii(tmpdir):
    path = tmpdir.strpath
    d = {'foo': numpy.random.randn(10, 10),
         'bar': numpy.random.randn(10, 10)}

    with WriteHelper('ark,t,f,scp:{p}/out.ark,{p}/out.scp'
                     .format(p=path)) as w:
        for k, v in d.items():
            w(k, v)
    from_ark = dict(load_ark('{p}/out.ark'.format(p=path)))
    from_scp = load_scp('{p}/out.scp'.format(p=path))
    _compare_allclose(from_ark, d)
    _compare_allclose(from_scp, d)


def test_segments(tmpdir):
    # Create wav.scp
    path = tmpdir.mkdir('test')
    wavscp = path.join('wav.scp').strpath

    rate = 500
    with open(wavscp, 'w') as f:
        wav = path.join('0.wav').strpath
        array0 = numpy.random.randint(0, 10, 2000, dtype=numpy.int16)
        write_wav(wav, rate, array0)
        f.write('wav0 {}\n'.format(wav))

        wav = path.join('1.wav').strpath
        array1 = numpy.random.randint(0, 10, 2000, dtype=numpy.int16)
        write_wav(wav, rate, array1)
        f.write('wav1 {}\n'.format(wav))

    # Create segments
    segments = path.join('segments').strpath
    with open(segments, 'w') as f:
        f.write('utt1 wav0 0.1 0.2\n')
        f.write('utt2 wav0 0.4 0.6\n')
        f.write('utt3 wav1 0.4 0.5\n')
        f.write('utt4 wav1 0.6 0.8\n')

    with ReadHelper('scp:{}'.format(wavscp), segments=segments) as r:
        d = {k: a for k, a in r}

        numpy.testing.assert_array_equal(
            d['utt1'][1], array0[int(0.1 * rate):int(0.2 * rate)])
        numpy.testing.assert_array_equal(
            d['utt2'][1], array0[int(0.4 * rate):int(0.6 * rate)])
        numpy.testing.assert_array_equal(
            d['utt3'][1], array1[int(0.4 * rate):int(0.5 * rate)])
        numpy.testing.assert_array_equal(
            d['utt4'][1], array1[int(0.6 * rate):int(0.8 * rate)])


def _compare(d1, d2):
    assert len(d1) != 0
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        numpy.testing.assert_array_equal(d1[key], d2[key])


def _compare_allclose(d1, d2, rtol=1e-07, atol=0.):
    assert len(d1) != 0
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        numpy.testing.assert_allclose(d1[key], d2[key], rtol, atol)
