from __future__ import unicode_literals

from contextlib import contextmanager
import io
from io import TextIOBase
import os
import subprocess
import sys
import warnings

PY3 = sys.version_info[0] == 3

if PY3:
    from collections.abc import MutableMapping
    string_types = str,
    text_type = str
else:
    from collections import MutableMapping
    string_types = basestring,  # noqa: F821
    text_type = unicode  # noqa: F821

default_encoding = 'utf-8'


"""
"utf-8" is used not depending on the environements variable,
e.g. LC_ALL, PYTHONIOENCODING, or PYTHONUTF8.

# Note: About the encoding of Python
- filesystem encoding

sys.getfilesystemencoding().
Used for file path and command line arguments.
The default value depends on the local in your unix system.
If Python>=3.7,

- preferred encoding

locale.getpreferredencoding(). Used for open().
The default value depends on the local in your unix system.

- stdout and stdin

If PYTHONIOENCODING is set, then it's used,
else if in a terminal, same as filesystem encoding
else same as preferred encoding

- default encoding

The default encoding for str.encode() or bytes.decode().
If Python2, it's ascii, if Python3, it's utf-8.

"""


if PY3:
    def my_popen(cmd, mode='r', buffering=-1):
        """Originated from python os module

        Extend for supporting mode == 'rb' and 'wb'

        Args:
            cmd (str):
            mode (str):
            buffering (int):
        """
        if isinstance(cmd, text_type):
            cmd = cmd.encode(default_encoding)
        if buffering == 0 or buffering is None:
            raise ValueError('popen() does not support unbuffered streams')
        if mode == 'r':
            proc = subprocess.Popen(cmd,
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    bufsize=buffering)
            return _wrap_close(io.TextIOWrapper(proc.stdout,
                                                encoding=default_encoding),
                               proc)
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
            return _wrap_close(io.TextIOWrapper(proc.stdin,
                                                encoding=default_encoding),
                               proc)
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

    def close(self):
        # Never close
        pass

    def __getattr__(self, name):
        return getattr(self.fd, name)

    def __iter__(self):
        return iter(self.fd)


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

    # If writting to stdout
    if name.strip().endswith('|'):
        cmd = name.strip()[:-1].encode(default_encoding)
        return my_popen(cmd, mode)
    # If reading from stdin
    elif name.strip().startswith('|'):
        cmd = name.strip()[1:].encode(default_encoding)
        return my_popen(cmd, mode)
    # If read mode
    elif name == '-' and 'r' in mode:
        if PY3:
            if mode == 'rb':
                return _stdstream_wrap(sys.stdin.buffer)
            else:
                return _stdstream_wrap(
                    io.TextIOWrapper(sys.stdin.buffer,
                                     encoding=default_encoding))
        else:
            return _stdstream_wrap(sys.stdin)
    # If write mode
    elif name == '-' and ('w' in mode or 'a' in mode):
        if PY3:
            if (mode == 'wb' or mode == 'ab'):
                return _stdstream_wrap(sys.stdout.buffer)
            else:
                return _stdstream_wrap(
                    io.TextIOWrapper(sys.stdout.buffer,
                                     encoding=default_encoding))
        else:
            return _stdstream_wrap(sys.stdout)
    else:
        encoding = None if 'b' in mode else default_encoding
        return io.open(name, mode, encoding=encoding)


@contextmanager
def open_or_fd(fname, mode):
    # If fname is a file name
    if isinstance(fname, string_types):
        encoding = None if 'b' in mode else default_encoding
        f = io.open(fname, mode, encoding=encoding)
    # If fname is a file descriptor
    else:
        if PY3 and 'b' in mode and isinstance(fname, TextIOBase):
            f = fname.buffer
        else:
            f = fname
    yield f

    if isinstance(fname, string_types):
        f.close()


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
        if not self.seekable():
            if PY3:
                raise OSError
            else:
                raise IOError
        if offset != 0:
            raise NotImplementedError('offset={}'.format(offset))
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
            if PY3:
                raise OSError
            else:
                raise IOError
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

    def readline(self, size=-1):
        remain = size
        string = None
        for f in self.fds:
            if string is None:
                string = f.readline(remain)
            else:
                string += f.readline(remain)
            if isinstance(string, text_type):
                if string.endswith("\n"):
                    break
            else:
                if string.endswith(b"\n"):
                    break
            remain = size - len(string)
            if remain == 0:
                break
            elif remain < 0:
                remain = -1
        return string



class CountFileDescriptor(object):
    def __init__(self, f):
        self.f = f
        self.position = 0

    def close(self):
        return self.f.close()

    def closed(self):
        return self.f.closed()

    def fileno(self):
        return self.f.flieno()

    def flush(self):
        return self.f.flush()

    def isatty(self):
        return self.f.isatty()

    def readbale(self):
        return self.f.readable()

    def readline(self, size=-1):
        line = self.f.readline(size)
        self.position += len(line)
        return line

    def readlines(self, hint=-1):
        lines = self.f.readlines(hint)
        for line in lines:
            self.position += len(line)
        return lines

    def seek(self, offset, whence=0):
        raise RuntimeError("Can't use seek")

    def seekable(self):
        return False

    def tell(self):
        return self.f.tell()

    def truncate(self, size=None):
        return self.f.trauncate(size)

    def writable(self):
        return self.f.writable()

    def writelines(self, lines):
        for line in lines:
            self.position += len(line)
        return self.f.writelines(lines)

    def __del__(self):
        return self.__del__()

    def read(self, size=-1):
        data = self.f.read(size)
        self.position += len(data)
        return data

    def readall(self):
        data = self.f.readall()
        self.position += len(data)
        return data

    def readinfo(self, b):
        nbyte = self.f.readinfo(b)
        self.position += nbyte
        return nbyte

    def write(self, b):
        self.position += b
        self.write(b)



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
