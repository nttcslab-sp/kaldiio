import sys

from kaldiio.utils import open_like_kaldi


def test_open_like_kaldi(tmpdir):
    with open_like_kaldi('echo hello |', 'r') as f:
        assert f.read() == 'hello\n'
    txt = tmpdir.mkdir('test').join('out.txt').strpath
    with open_like_kaldi('| cat > {}'.format(txt), 'w') as f:
        f.write('hello')
    with open(txt, 'r') as f:
        assert f.read() == 'hello'


def test_open_stdsteadm():
    with open_like_kaldi('-', 'w') as f:
        assert f is sys.stdout
    with open_like_kaldi('-', 'wb'):
        pass
    with open_like_kaldi('-', 'r') as f:
        assert f is sys.stdin
    with open_like_kaldi('-', 'rb'):
        pass
