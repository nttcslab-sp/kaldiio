import io
import os
import subprocess
import sys
from contextlib import contextmanager
from io import TextIOBase

from six import string_types

PY3 = sys.version_info[0] == 3


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

    A proxy for a file whose close waits for the process"""
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


@contextmanager
def _stdstream_wrap(fd):
    yield fd


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

    name = name.strip()
    if name[-1] == '|':
        return my_popen(name[:-1], mode)
    elif name[0] == '|':
        return my_popen(name[1:], mode)
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
