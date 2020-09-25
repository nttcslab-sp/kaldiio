# coding: utf-8
from __future__ import unicode_literals

from kaldiio.utils import LimitedSizeDict


def test_limted_size_dict():
    d = LimitedSizeDict(3)
    d["foo"] = 1
    d["bar"] = 2
    d["baz"] = 3

    assert "foo" in d
    assert "bar" in d
    assert "baz" in d

    d["foo2"] = 4
    assert "foo" not in d
    assert "foo2" in d

    d["bar2"] = 4
    assert "bar" not in d
    assert "bar2" in d
