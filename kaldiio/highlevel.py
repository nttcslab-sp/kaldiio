from collections import OrderedDict

from .arkio import load_ark
from .arkio import load_scp
from .arkio import save_ark
from .utils import open_like_kaldi


def parse_specifier(specifier):
    """A utility to parse "specifier"

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
            raise ValueError('{} is duplicated.'.format(x))

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
    """A heghlevel interface to write ark or/and scp

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


class ReadHelper(object):
    """A highlevel interface to load ark or scp

    >>> import numpy
    >>> array_in = numpy.random.randn(10, 10)
    >>> save_ark('feats.ark', {'foo': array_in}, scp='feats.scp')
    >>> helper = ReadHelper('ark:cat feats.ark |')
    >>> for uttid, array_out in helper:
    ...     assert uttid == 'foo'
    ...     numpy.testing.assert_array_equal(array_in, array_out)
    >>> helper = ReadHelper('scp:feats.scp')
    >>> for uttid, array_out in helper:
    ...     assert uttid == 'foo'
    ...     numpy.testing.assert_array_equal(array_in, array_out)

    """
    def __init__(self, wspecifier):
        spec_dict = parse_specifier(wspecifier)
        if len(spec_dict) != 1:
            raise RuntimeError('Specify one of scp or ark in rspecifier')

        if 'scp' in spec_dict:
            self.scp = True
            mode = 'r'
        else:
            self.scp = False
            mode = 'rb'

        if self.scp:
            with open_like_kaldi(next(iter(spec_dict.values())), mode) as f:
                self.dict = load_scp(f)
            self.file = None
        else:
            self.dict = None
            self.file = open_like_kaldi(next(iter(spec_dict.values())), mode)
        self.closed = False

    def __iter__(self):
        if self.scp:
            for k, v in self.dict.items():
                yield k, v
        else:
            with self.file as f:
                for k, v in load_ark(f):
                    yield k, v
            self.closed = True

    def __len__(self):
        if not self.scp:
            raise RuntimeError('__getitem__() is supported only when scp mode')
        return len(self.dict)

    def __contains__(self, item):
        if not self.scp:
            raise RuntimeError(
                '__contains__() is supported only when scp mode')
        return item in self.dict

    def __getitem__(self, item):
        if not self.scp:
            raise RuntimeError('__getitem__() is supported only when scp mode')
        return self.dict[item]

    def get(self, item, default=None):
        if not self.scp:
            raise RuntimeError('get() is supported only when scp mode')
        return self.dict.get(item, default)

    def keys(self):
        if not self.scp:
            raise RuntimeError('keys() is supported only when scp mode')
        return self.dict.keys()

    def items(self):
        if not self.scp:
            raise RuntimeError('items() is supported only when scp mode')
        return self.dict.items()

    def values(self):
        if not self.scp:
            raise RuntimeError('values() is supported only when scp mode')
        return self.dict.values()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.scp and not self.closed:
            self.file.close()

    def __del__(self):
        if not self.scp and not self.closed:
            self.close()

    def close(self):
        if not self.scp and not self.closed:
            self.file.close()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
