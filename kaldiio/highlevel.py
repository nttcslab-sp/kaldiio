from collections.__init__ import OrderedDict

from kaldiio.arkio import save_ark
from kaldiio.utils import open_like_kaldi


def parse_specifier(specifier):
    """

    Args:
        specifier (str):
    Returns:
        parsed_dict (OrderedDict):
            Like {'ark': 'file.ark', 'scp': 'file.scp'}


    >>> d = parse_specifier('ark,t,scp:file.ark,file.scp')
    >>> print(d['ark,t'])
    file.ark

    """
    if not isinstance(specifier, str):
        raise TypeError(
            'Argument must be str, but got {}'.format(type(specifier)))
    sp = specifier.split(':', 1)
    if len(sp) != 2:
        if ':' not in specifier:
            raise ValueError('The output file must be specified with '
                             'kaldi-specifier style,'
                             ' e.g. ark,scp:out.ark,out.scp, but you gave as '
                             '{}'.format(specifier))

    types, files = sp
    types = list((map(lambda x: x.strip(), types.split(','))))
    files = list((map(lambda x: x.strip(), files.split(','))))
    for x in set(types):
        if types.count(x) > 1:
            raise ValueError(f'{x} is duplicated.')

    supported = [{'ark'}, {'scp'}, {'ark', 'scp'},
                 {'ark', 't'}, {'scp', 'ark', 't'}]
    if set(types) not in supported:
        raise ValueError(
            'Invalid type: {}, must be one of {}'.format(types, supported))

    if 't' in types:
        types.remove('t')
        types[types.index('ark')] = 'ark,t'

    if len(types) != len(files):
        raise ValueError(
            'The number of file types need to match with the file names: '
            '{} != {}, you gave as {}'.format(len(types), len(files),
                                              specifier))

    return OrderedDict(zip(types, files))


class WriteHelper(object):
    """

    >>> helper = WriteHelper('ark,scp:a.ark,b.ark')
    >>> helper('uttid', array)

    """
    def __init__(self, wspecifier):
        spec_dict = parse_specifier(wspecifier)
        if set(spec_dict) == {'scp'}:
            raise ValueError(
                'Writing only in a scp file is not supported. '
                'Please specify a ark file with a scp file.')

        if 'ark,t' in spec_dict:
            ark_file = spec_dict['ark,t']
            self.text = True
        else:
            ark_file = spec_dict['ark']
            self.text = False

        self.fark = open_like_kaldi(ark_file, 'wb')
        if 'scp' in spec_dict:
            self.fscp = open_like_kaldi(spec_dict['scp'], 'w')
        else:
            self.fscp = None
        self.closed = False

    def __call__(self, key, array):
        if self.closed:
            raise RuntimeError('WriteHelper has been already closed')
        save_ark(self.fark, {key: array}, scp=self.fscp, text=self.text)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fark.close()
        if self.fscp is not None:
            self.fscp.close()

    def __del__(self):
        if not self.closed:
            self.close()

    def close(self):
        self.fark.close()
        if self.fscp is not None:
            self.fscp.close()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
