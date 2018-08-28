# Kaldiio
A pure python moudle for reading and writing kaldi ark files


## Dependencies

    numpy
    scipy
    six

## Usage
### Basic

```python
import kaldio

d = kaldiio.load_ark('a.ark')  # d is a generator object
for key, array in d:
    ...

# === load_scp can load ark file as lazy dict
d = kaldiio.load_scp('a.scp')
for key in d:
    d[key]

# === Create ark file from numpy
kaldiio.save_ark('b.ark', {'key': array, 'key2': array2})
# Create ark with scp _file, too
kaldiio.save_ark('b.ark', {'key': array, 'key2': array2},
                 scp='b.scp')

# === load_ark and load_scp can accepts file descriptor, too
with open('a.ark') as fd:
    kaldiio.load_ark(fd)
with open('a.scp') as fd:
    kaldiio.load_scp(fd)

# === Writes arrays to sys.stdout
import sys
kaldiio.save_ark(sys.stdout, {'key': array})

# === Writes arrays for each keys
# generate a.ark
kaldiio.save_ark('a.ark', {'key': array})
# After here, a.ark is opened with 'a' (append) mode.
kaldiio.save_ark('a.ark', {'key2': array2}, append=True)
```

## Utility
### open_like_kaldi

``kaldiio.open_like_kaldi`` maybe a useful tool if you are familiar with Kaldi. This functions can performs as following,

```python
from kaldiio import open_like_kaldi
with open_like_kaldi('echo -n hello |', 'r') as f:
    assert f.read() == 'hello'
with open_like_kaldi('| cat > out.txt', 'w') as f:
    f.write('hello')
with open('out.txt', 'r') as f:
    assert f.read() == 'hello'

import sys
with open_like_kaldi('-', 'r') as f:
    assert f is sys.stdin
with open_like_kaldi('-', 'w') as f:
    assert f is sys.stdout
```

For example, there are gziped alignment file, how to open it:
```python
from kaldiio import open_like_kaldi, load_ark
with open_like_kaldi('gzip -c exp/tri3_ali/ali.*.gz |', 'rb') as f:
    # Alignment format equals ark of IntVector
    g = load_ark(f)
    for k, array in g:
        ...
```

## parse_specifier

```python
from kaldiio import parse_specifier, open_like_kaldi, load_ark
rspecifier = 'ark:gunzip -c file.ark.gz |'
spec_dict = parse_specifier(rspecifier)
# spec_dict = {'ark': 'gunzip -c file.ark.gz |'}

for open_like_kaldi(spec_dict['ark'], 'rb') as fark:
    for key, array in load_ark(fark):
        ...
```

### WriteHelper
This is a high level module for writing in the similar style to Kaldi.

```python
import numpy
from kaldiio import WriteHelper
wspecifier = 'ark,scp:file.ark,file.scp'
# You can also use pipe form
# wspecifier = 'ark:| gzip -c > file.gz'

with WriteHelper(wspecifier) as writer:
    for i in range(10):
        writer(str(i), numpy.random.randn(10, 10))
```
