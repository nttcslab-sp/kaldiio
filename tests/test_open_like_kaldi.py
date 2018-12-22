import sys

import kaldiio


def test_open_like_kaldi(tmpdir):
    with kaldiio.open_like_kaldi('echo -n hello |', 'r') as f:
        assert f.read() == 'hello'
    txt = tmpdir.mkdir('test').join('out.txt').strpath
    with kaldiio.open_like_kaldi('| cat > {}'.format(txt), 'w') as f:
        f.write('hello')
    with open(txt, 'r') as f:
        assert f.read() == 'hello'


def test_open_stdsteadm():
    with kaldiio.open_like_kaldi('-', 'w') as f:
        assert f is sys.stdout
    with kaldiio.open_like_kaldi('-', 'wb'):
        pass
    with kaldiio.open_like_kaldi('-', 'r') as f:
        assert f is sys.stdin
    with kaldiio.open_like_kaldi('-', 'rb'):
        pass
