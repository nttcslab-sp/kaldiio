# coding: utf-8
from __future__ import unicode_literals

import glob
import io
import os

import numpy as np
import pytest

import kaldiio
from kaldiio.matio import _parse_arkpath

arkdir = os.path.join(os.path.dirname(__file__), "arks")


@pytest.mark.parametrize("fname", glob.glob(os.path.join(arkdir, "*.ark")))
def test_read_arks(fname):
    # Assume arks dir existing at the same directory
    ark0 = dict(
        kaldiio.load_ark(os.path.join(os.path.dirname(__file__), "arks", "test.ark"))
    )
    ark = dict(kaldiio.load_ark(fname))
    _compare_allclose(ark, ark0, atol=1e-1)


@pytest.mark.parametrize(
    "shape1,shape2", [[(1000, 120), (10, 120)], [(0, 0), (0, 120)], [(100,), (120,)]]
)
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("max_cache_fd", [0, 3])
def test_write_read(tmpdir, shape1, shape2, endian, dtype, max_cache_fd):
    path = tmpdir.mkdir("test")

    a = np.random.rand(*shape1).astype(dtype)
    b = np.random.rand(*shape2).astype(dtype)
    origin = {"Ï,é,à": a, "あいうえお": b}
    kaldiio.save_ark(
        path.join("a.ark").strpath,
        origin,
        scp=path.join("b.scp").strpath,
        endian=endian,
    )

    d2 = {k: v for k, v in kaldiio.load_ark(path.join("a.ark").strpath, endian=endian)}
    d5 = {
        k: v
        for k, v in kaldiio.load_scp(
            path.join("b.scp").strpath, endian=endian, max_cache_fd=max_cache_fd
        ).items()
    }
    with io.open(path.join("a.ark").strpath, "rb") as fd:
        d6 = {k: v for k, v in kaldiio.load_ark(fd, endian=endian)}
    _compare(d2, origin)
    _compare(d5, origin)
    _compare(d6, origin)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_read_multiark(tmpdir, endian, dtype):
    path = tmpdir.mkdir("test")

    a = np.random.rand(1000, 120).astype(dtype)
    b = np.random.rand(10, 120).astype(dtype)
    origin = {"Ï,é,à": a, "あいうえお": b}

    kaldiio.save_ark(
        path.join("a.ark").strpath,
        origin,
        scp=path.join("b.scp").strpath,
        endian=endian,
    )

    c = np.random.rand(1000, 120).astype(dtype)
    d = np.random.rand(10, 120).astype(dtype)
    origin.update({"c": c, "d": d})
    with io.open(path.join("b.scp").strpath, "a", encoding="utf-8") as f:
        kaldiio.save_ark(path.join("b.ark").strpath, origin, scp=f, endian=endian)

    d5 = {
        k: v
        for k, v in kaldiio.load_scp(path.join("b.scp").strpath, endian=endian).items()
    }
    _compare(d5, origin)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_write_read_sequential(tmpdir, endian):
    path = tmpdir.mkdir("test")

    a = np.random.rand(1000, 120).astype(np.float32)
    b = np.random.rand(10, 120).astype(np.float32)
    origin = {"Ï,é,à": a, "あいうえお": b}
    kaldiio.save_ark(
        path.join("a.ark").strpath,
        origin,
        scp=path.join("b.scp").strpath,
        endian=endian,
    )

    d5 = {
        k: v
        for k, v in kaldiio.load_scp_sequential(
            path.join("b.scp").strpath, endian=endian
        )
    }
    _compare(d5, origin)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_write_read_multiark_sequential(tmpdir, endian):
    path = tmpdir.mkdir("test")

    a = np.random.rand(1000, 120).astype(np.float32)
    b = np.random.rand(10, 120).astype(np.float32)
    origin = {"Ï,é,à": a, "あいうえお": b}

    kaldiio.save_ark(
        path.join("a.ark").strpath,
        origin,
        scp=path.join("b.scp").strpath,
        endian=endian,
    )

    c = np.random.rand(1000, 120).astype(np.float32)
    d = np.random.rand(10, 120).astype(np.float32)
    origin.update({"c": c, "d": d})
    with io.open(path.join("b.scp").strpath, "a", encoding="utf-8") as f:
        kaldiio.save_ark(path.join("b.ark").strpath, origin, scp=f, endian=endian)

    d5 = {
        k: v
        for k, v in kaldiio.load_scp_sequential(
            path.join("b.scp").strpath, endian=endian
        )
    }
    _compare(d5, origin)


def test_write_read_ascii(tmpdir):
    path = tmpdir.mkdir("test")
    a = np.random.rand(10, 10).astype(np.float32)
    b = np.random.rand(5, 35).astype(np.float32)
    origin = {"Ï,é,à": a, "あいうえお": b}
    kaldiio.save_ark(
        path.join("a.ark").strpath, origin, scp=path.join("a.scp").strpath, text=True
    )
    d2 = {k: v for k, v in kaldiio.load_ark(path.join("a.ark").strpath)}
    d5 = {k: v for k, v in kaldiio.load_scp(path.join("a.scp").strpath).items()}
    _compare_allclose(d2, origin)
    _compare_allclose(d5, origin)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_write_read_int32_vector(tmpdir, endian):
    path = tmpdir.mkdir("test")

    a = np.random.randint(1, 128, 10, dtype=np.int32)
    b = np.random.randint(1, 128, 10, dtype=np.int32)
    origin = {"Ï,é,à": a, "あいうえお": b}
    kaldiio.save_ark(
        path.join("a.ark").strpath,
        origin,
        scp=path.join("b.scp").strpath,
        endian=endian,
    )

    d2 = {k: v for k, v in kaldiio.load_ark(path.join("a.ark").strpath, endian=endian)}
    d5 = {
        k: v
        for k, v in kaldiio.load_scp(path.join("b.scp").strpath, endian=endian).items()
    }
    with io.open(path.join("a.ark").strpath, "rb") as fd:
        d6 = {k: v for k, v in kaldiio.load_ark(fd, endian=endian)}
    _compare(d2, origin)
    _compare(d5, origin)
    _compare(d6, origin)


def test_write_read_int32_vector_ascii(tmpdir):
    path = tmpdir.mkdir("test")

    a = np.random.randint(1, 128, 10, dtype=np.int32)
    b = np.random.randint(1, 128, 10, dtype=np.int32)
    origin = {"Ï,é,à": a, "あいうえお": b}
    kaldiio.save_ark(
        path.join("a.ark").strpath, origin, scp=path.join("b.scp").strpath, text=True
    )

    d2 = {k: v for k, v in kaldiio.load_ark(path.join("a.ark").strpath)}
    d5 = {k: v for k, v in kaldiio.load_scp(path.join("b.scp").strpath).items()}
    with io.open(path.join("a.ark").strpath, "rb") as fd:
        d6 = {k: v for k, v in kaldiio.load_ark(fd)}
    _compare_allclose(d2, origin)
    _compare_allclose(d5, origin)
    _compare_allclose(d6, origin)


@pytest.mark.parametrize("compression_method", [1, 3, 5])
def test_write_compressed_arks(tmpdir, compression_method):
    # Assume arks dir existing at the same directory
    ark0 = dict(
        kaldiio.load_ark(os.path.join(os.path.dirname(__file__), "arks", "test.ark"))
    )
    path = tmpdir.mkdir("test").join("c.ark").strpath
    kaldiio.save_ark(path, ark0, compression_method=compression_method)
    arkc = dict(kaldiio.load_ark(path))
    arkc_valid = dict(
        kaldiio.load_ark(
            os.path.join(
                os.path.dirname(__file__),
                "arks",
                "test.cm{}.ark".format(compression_method),
            )
        )
    )
    _compare_allclose(arkc, arkc_valid, atol=1e-4)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compression_method", [2, 3, 7])
def test_write_read_compress(tmpdir, compression_method, endian):
    path = tmpdir.mkdir("test")

    a = np.random.rand(1000, 120).astype(np.float32)
    b = np.random.rand(10, 120).astype(np.float32)
    origin = {"Ï,é,à": a, "あいうえお": b}
    kaldiio.save_ark(
        path.join("a.ark").strpath,
        origin,
        scp=path.join("b.scp").strpath,
        compression_method=compression_method,
        endian=endian,
    )

    d2 = {k: v for k, v in kaldiio.load_ark(path.join("a.ark").strpath, endian=endian)}
    d5 = {
        k: v
        for k, v in kaldiio.load_scp(path.join("b.scp").strpath, endian=endian).items()
    }
    with io.open(path.join("a.ark").strpath, "rb") as fd:
        d6 = {k: v for k, v in kaldiio.load_ark(fd, endian=endian)}
    _compare_allclose(d2, origin, atol=1e-1)
    _compare_allclose(d5, origin, atol=1e-1)
    _compare_allclose(d6, origin, atol=1e-1)


def test_append_mode(tmpdir):
    path = tmpdir.mkdir("test")

    a = np.random.rand(1000, 120).astype(np.float32)
    b = np.random.rand(10, 120).astype(np.float32)
    origin = {"Ï,é,à": a, "あいうえお": b}
    kaldiio.save_ark(path.join("a.ark").strpath, origin, scp=path.join("b.scp").strpath)

    kaldiio.save_ark(
        path.join("a2.ark").strpath,
        {"Ï,é,à": a},
        scp=path.join("b2.scp").strpath,
        append=True,
    )
    kaldiio.save_ark(
        path.join("a2.ark").strpath,
        {"あいうえお": b},
        scp=path.join("b2.scp").strpath,
        append=True,
    )
    d1 = {k: v for k, v in kaldiio.load_ark(path.join("a.ark").strpath)}
    d2 = {k: v for k, v in kaldiio.load_scp(path.join("b.scp").strpath).items()}
    d3 = {k: v for k, v in kaldiio.load_ark(path.join("a2.ark").strpath)}
    d4 = {k: v for k, v in kaldiio.load_scp(path.join("b2.scp").strpath).items()}
    _compare(d1, origin)
    _compare(d2, origin)
    _compare(d3, origin)
    _compare(d4, origin)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_read_mat(tmpdir, endian, dtype):
    path = tmpdir.mkdir("test")
    valid = np.random.rand(1000, 120).astype(dtype)
    kaldiio.save_mat(path.join("a.mat").strpath, valid, endian=endian)
    test = kaldiio.load_mat(path.join("a.mat").strpath, endian=endian)
    np.testing.assert_array_equal(test, valid)


def test__parse_arkpath():
    assert _parse_arkpath("a.ark") == ("a.ark", None, None)
    assert _parse_arkpath("a.ark:12") == ("a.ark", 12, None)
    assert _parse_arkpath("a.ark:12[4]") == ("a.ark", 12, (slice(4, 5, None),))
    assert _parse_arkpath("a.ark:12[3:4]") == ("a.ark", 12, (slice(3, 5, None),))
    assert _parse_arkpath("a.ark:12[3:10:2]") == ("a.ark", 12, (slice(3, 11, 2),))
    assert _parse_arkpath("a.ark:12[2:6,3:4]") == (
        "a.ark",
        12,
        (slice(2, 7), slice(3, 5, None)),
    )
    assert _parse_arkpath('cat "fo:o.ark" |') == ('cat "fo:o.ark" |', None, None)


def _compare(d1, d2):
    assert len(d1) != 0
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        np.testing.assert_array_equal(d1[key], d2[key])


def _compare_allclose(d1, d2, rtol=1e-07, atol=0.0):
    assert len(d1) != 0
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        np.testing.assert_allclose(d1[key], d2[key], rtol, atol)


if __name__ == "__main__":
    pytest.main()
