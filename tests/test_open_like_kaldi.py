# coding: utf-8
from __future__ import unicode_literals

import io
import sys

from kaldiio.utils import open_like_kaldi

PY3 = sys.version_info[0] == 3


def test_open_like_kaldi(tmpdir):
    with open_like_kaldi("echo あああ |", "r") as f:
        if PY3:
            assert f.read() == "あああ\n"
        else:
            assert f.read().decode("utf-8") == "あああ\n"
    txt = tmpdir.mkdir("test").join("out.txt").strpath
    with open_like_kaldi("| cat > {}".format(txt), "w") as f:
        if PY3:
            f.write("あああ")
        else:
            f.write("あああ".encode("utf-8"))
    with io.open(txt, "r", encoding="utf-8") as f:
        assert f.read() == "あああ"
