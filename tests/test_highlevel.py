import numpy

from kaldiio.highlevel import ReadHelper
from kaldiio.highlevel import WriteHelper
from kaldiio.matio import load_ark
from kaldiio.matio import load_scp
from kaldiio.matio import save_ark


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


def test_write_helper(tmpdir):
    path = tmpdir.strpath
    d = {'foo': numpy.random.randn(10, 10),
         'bar': numpy.random.randn(10, 10)}

    with WriteHelper('ark,scp:{p}/out.ark, {p}/out.scp'.format(p=path)) as w:
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

    with WriteHelper('ark,t,scp:{p}/out.ark, {p}/out.scp'.format(p=path)) as w:
        for k, v in d.items():
            w(k, v)
    from_ark = dict(load_ark('{p}/out.ark'.format(p=path)))
    from_scp = load_scp('{p}/out.scp'.format(p=path))
    _compare_allclose(from_ark, d)
    _compare_allclose(from_scp, d)


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
