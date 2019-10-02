import pytest

from kaldiio import parse_specifier


def test_ark():
    d = parse_specifier('ark:file.ark')
    assert d['ark'] == 'file.ark'


def test_scp():
    d = parse_specifier('scp:file.scp')
    assert d['scp'] == 'file.scp'


def test_ark_scp():
    d = parse_specifier('ark,scp:file.ark,file.scp')
    assert d['ark'] == 'file.ark'
    assert d['scp'] == 'file.scp'


def test_scp_ark():
    d = parse_specifier('scp,ark:file.scp,file.ark')
    assert d['ark'] == 'file.ark'
    assert d['scp'] == 'file.scp'


def test_error1():
    with pytest.raises(ValueError):
        parse_specifier('ffdafafaf')


def test_error2():
    with pytest.raises(ValueError):
        parse_specifier('ak,sp:file.ark,file.scp')


def test_error3():
    with pytest.raises(ValueError):
        parse_specifier('ark,scp:file.ark')


def test_error4():
    with pytest.raises(ValueError):
        parse_specifier('ark,ark:file.ark,file.ark')
