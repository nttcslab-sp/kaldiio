import warnings

from kaldiio.matio import load_ark
from kaldiio.matio import load_scp
from kaldiio.matio import save_ark
from kaldiio.utils import open_like_kaldi
from kaldiio.utils import parse_specifier


class WriteHelper(object):
    """A heghlevel interface to write ark or/and scp

    >>> helper = WriteHelper('ark,scp:a.ark,b.ark')
    >>> helper('uttid', array)

    """
    def __init__(self, wspecifier, compression_method=None):
        self.initialized = False
        self.closed = False

        self.compression_method = compression_method
        spec_dict = parse_specifier(wspecifier)
        if spec_dict['scp'] is not None and spec_dict['ark'] is None:
            raise ValueError(
                'Writing only in a scp file is not supported. '
                'Please specify a ark file with a scp file.')
        for k in spec_dict:
            if spec_dict[k] and k not in ('scp', 'ark', 't', 'f'):
                warnings.warn(
                    '{} option is given, but currently it never affects'
                    .format(k))

        self.text = spec_dict['t']
        self.flush = spec_dict['f']
        ark_file = spec_dict['ark']
        self.fark = open_like_kaldi(ark_file, 'wb')
        if spec_dict['scp'] is not None:
            self.fscp = open_like_kaldi(spec_dict['scp'], 'w')
        else:
            self.fscp = None
        self.initialized = True

    def __call__(self, key, array):
        if self.closed:
            raise RuntimeError('WriteHelper has been already closed')
        save_ark(self.fark, {key: array}, scp=self.fscp, text=self.text,
                 compression_method=self.compression_method)

        if self.flush:
            if self.fark is not None:
                self.fark.flush()
            if self.fscp is not None:
                self.fscp.flush()

    def __setitem__(self, key, value):
        self(key, value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.initialized and not self.closed:
            self.fark.close()
            if self.fscp is not None:
                self.fscp.close()
            self.closed = True


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
    def __init__(self, wspecifier, segments=None):
        self.initialized = False
        self.scp = None
        self.closed = False

        spec_dict = parse_specifier(wspecifier)
        if spec_dict['scp'] is not None and spec_dict['ark'] is not None:
            raise RuntimeError('Specify one of scp or ark in rspecifier')
        for k in spec_dict:
            if spec_dict[k] and k not in ('scp', 'ark', 'p'):
                warnings.warn(
                    '{} option is given, but currently it never affects'
                    .format(k))
        self.permissive = spec_dict['p']

        if spec_dict['scp'] is not None:
            self.scp = spec_dict['scp']
        else:
            self.scp = False

        if self.scp:
            with open_like_kaldi(spec_dict['scp'], 'r') as f:
                self.dict = load_scp(f)

            self.file = None
        else:
            self.dict = None
            self.file = open_like_kaldi(spec_dict['ark'], 'rb')
        self.initialized = True

    def __iter__(self):
        if self.scp:
            it = iter(self.dict.items())
            while True:
                try:
                    k, v = next(it)
                except StopIteration:
                    break
                except Exception:
                    if self.permissive:
                        # Continue if error happen
                        continue
                    else:
                        raise
                yield k, v
        else:
            with self.file as f:
                gen = load_ark(f)
                while True:
                    try:
                        k, v = next(gen)
                    except StopIteration:
                        break
                    except Exception:
                        if self.permissive:
                            # Stop if error happen
                            break
                        else:
                            raise
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

    def close(self):
        if self.initialized and not self.scp and not self.closed:
            self.file.close()
            self.closed = True
