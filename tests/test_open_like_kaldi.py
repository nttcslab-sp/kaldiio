# coding: utf-8
import io

from kaldiio.utils import open_like_kaldi


def test_open_like_kaldi(tmpdir):
    with open_like_kaldi('echo あああ |', 'r') as f:
        assert f.read() == 'あああ\n'
    txt = tmpdir.mkdir('test').join('out.txt').strpath
    with open_like_kaldi('| cat > {}'.format(txt), 'w') as f:
        f.write('あああ')
    with io.open(txt, 'r', encoding='utf-8') as f:
        assert f.read() == 'あああ'
