from contextlib import contextmanager
import io
from io import TextIOBase
import os
import subprocess
import sys
import warnings

from six import string_types

PY3 = sys.version_info[0] == 3

if PY3:
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping


if PY3:
    def my_popen(cmd, mode='r', buffering=-1):
        """Originated from python os module

        Extend for supporting mode == 'rb' and 'wb'

        Args:
            cmd (str):
            mode (str):
            buffering (int):
        """
        if not isinstance(cmd, str):
            raise TypeError(
                'invalid cmd type (%s, expected string)' % type(cmd))
        if buffering == 0 or buffering is None:
            raise ValueError('popen() does not support unbuffered streams')
        if mode == 'r':
            proc = subprocess.Popen(cmd,
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    bufsize=buffering)
            return _wrap_close(io.TextIOWrapper(proc.stdout), proc)
        elif mode == 'rb':
            proc = subprocess.Popen(cmd,
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    bufsize=buffering)
            return _wrap_close(proc.stdout, proc)
        elif mode == 'w':
            proc = subprocess.Popen(cmd,
                                    shell=True,
                                    stdin=subprocess.PIPE,
                                    bufsize=buffering)
            return _wrap_close(io.TextIOWrapper(proc.stdin), proc)
        elif mode == 'wb':
            proc = subprocess.Popen(cmd,
                                    shell=True,
                                    stdin=subprocess.PIPE,
                                    bufsize=buffering)
            return _wrap_close(proc.stdin, proc)
        else:
            raise TypeError('Unsupported mode == {}'.format(mode))
else:
    my_popen = os.popen


class _wrap_close(object):
    """Originated from python os module

    A proxy for a file whose close waits for the process
    """
    def __init__(self, stream, proc):
        self._stream = stream
        self._proc = proc

    def close(self):
        self._stream.close()
        returncode = self._proc.wait()
        if returncode == 0:
            return None
        if os.name == 'nt':
            return returncode
        else:
            return returncode << 8  # Shift left to match old behavior

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __iter__(self):
        return iter(self._stream)


class _stdstream_wrap(object):
    def __init__(self, fd):
        self.fd = fd

    def __enter__(self):
        return self.fd

    def __exit__(self, *args):
        # Never close
        pass

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __iter__(self):
        return iter(self._stream)


def open_like_kaldi(name, mode='r'):
    """Open a file like kaldi io

    Args:
        name (str or file):
        mode (str):
    """
    # If file descriptor
    if not isinstance(name, string_types):
        if PY3 and 'b' in mode and isinstance(name, TextIOBase):
            return name.buffer
        else:
            return name

    if name.strip()[-1] == '|':
        return my_popen(name.strip()[:-1], mode)
    elif name.strip()[0] == '|':
        return my_popen(name.strip()[1:], mode)
    elif name == '-' and 'r' in mode:
        if mode == 'rb' and PY3:
            return _stdstream_wrap(sys.stdin.buffer)
        else:
            return _stdstream_wrap(sys.stdin)
    elif name == '-' and ('w' in mode or 'a' in mode):
        if (mode == 'wb' or mode == 'ab') and PY3:
            return _stdstream_wrap(sys.stdout.buffer)
        else:
            return _stdstream_wrap(sys.stdout)
    else:
        return open(name, mode)


@contextmanager
def open_or_fd(fname, mode):
    # If fname is a file name
    if isinstance(fname, string_types):
        f = open(fname, mode)
    # If fname is a file descriptor
    else:
        if PY3 and 'b' in mode and isinstance(fname, TextIOBase):
            f = fname.buffer
        else:
            f = fname
    yield f

    if isinstance(fname, string_types):
        f.close()


def convert_to_slice(string):
    slices = []
    for ele in string.split(','):
        if ele == '' or ele == ':':
            slices.append(slice(None))
        else:
            args = []
            for _ele in ele.split(':'):
                if _ele == '':
                    args.append(None)
                else:
                    args.append(int(_ele))
            slices.append(slice(*args))
    return tuple(slices)


class MultiFileDescriptor(object):
    """What is this class?

    First of all, I want to load all format kaldi files
    only by using read_kaldi function, and I want to load it
    from file and file descriptor including standard input stream.
    To judge its file format it is required to make the
    file descriptor read and seek(to return original position).
    However, stdin is not seekable, so I create this clas.
    This class joints multiple file descriptors
    and I assume this class is used as follwoing,

        >>> string = fd.read(size)
        >>> # To check format from string
        >>> _fd = StringIO(string)
        >>> newfd = MultiFileDescriptor(_fd, fd)
    """
    def __init__(self, *fds):
        self.fds = fds

        if self.seekable():
            self.init_pos = [f.tell() for f in self.fds]
        else:
            self.init_pos = None

    def seek(self, offset, from_what=0):
        if offset != 0:
            raise NotImplementedError('offset={}'.format(offset))
        if not self.seekable():
            raise OSError
        if from_what == 1:
            offset += self.tell()
            from_what = 0

        if from_what == 0:
            for idx, f in enumerate(self.fds):
                pos = self.init_pos[idx]
                f.seek(pos + offset, 0)
                offset -= (f.tell() - pos)
        else:
            raise NotImplementedError('from_what={}'.format(from_what))

    def seekable(self):
        return all(seekable(f) for f in self.fds)

    def tell(self):
        if not self.seekable():
            raise OSError
        return sum(f.tell() - self.init_pos[idx]
                   for idx, f in enumerate(self.fds))

    def read(self, size=-1):
        remain = size
        string = None
        for f in self.fds:
            if string is None:
                string = f.read(remain)
            else:
                string += f.read(remain)
            remain = size - len(string)
            if remain == 0:
                break
            elif remain < 0:
                remain = -1
        return string


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
    sp = specifier.split(':', 1)
    if len(sp) != 2:
        if ':' not in specifier:
            raise ValueError('The output file must be specified with '
                             'kaldi-specifier style,'
                             ' e.g. ark,scp:out.ark,out.scp, but you gave as '
                             '{}'.format(specifier))

    types, files = sp
    types = types.split(',')
    if 'ark' not in types and 'scp' not in types:
        raise ValueError(
            'One of/both ark and scp is required: '
            'e.g. ark,scp:out.ark,out.scp: '
            '{}'.format(specifier))
    elif 'ark' in types and 'scp' in types:
        if ',' not in files:
            raise ValueError(
                'You specified both ark and scp, '
                'but a file path is given: '
                'e.g. ark,scp:out.ark,out.scp: {}'.format(specifier))
        files = files.split(',', 1)
    else:
        files = [files]

    spec_dict = {'ark': None,
                 'scp': None,
                 't': False,  # text
                 'o': False,  # once
                 'p': False,  # permissive
                 'f': False,  # flush
                 's': False,  # sorted
                 'cs': False,  # called-sorted
                 }
    for t in types:
        if t not in spec_dict:
            raise ValueError('Unknown option {}()'.format(t, types))
        if t in ('scp', 'ark'):
            if spec_dict[t] is not None:
                raise ValueError('You specified {} twice'.format(t))
            spec_dict[t] = files.pop(0)
        else:
            spec_dict[t] = True

    return spec_dict


class LazyLoader(MutableMapping):
    """Don't use this class directly"""
    def __init__(self, loader):
        self._dict = {}
        self._loader = loader

    def __repr__(self):
        return 'LazyLoader [{} keys]'.format(len(self))

    def __getitem__(self, key):
        ark_name = self._dict[key]
        try:
            return self._loader(ark_name)
        except Exception:
            warnings.warn(
                'An error happens at loading "{}"'.format(ark_name))
            raise

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return len(self._dict)

    def __contains__(self, item):
        return item in self._dict


def seekable(f):
    if hasattr(f, 'seekable'):
        return f.seekable()

    # For Py2
    else:
        if hasattr(f, 'tell'):
            try:
                f.tell()
            except (IOError, OSError):
                return False
            else:
                return True
        else:
            return False
