# Kaldiio
[![pypi](https://img.shields.io/pypi/v/kaldiio.svg)](https://pypi.python.org/pypi/kaldiio)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/kaldiio.svg)](https://pypi.python.org/pypi/kaldiio)
[![Build Status](https://travis-ci.com/nttcslab-sp/kaldiio.svg?branch=master)](https://travis-ci.com/nttcslab-sp/kaldiio)
[![codecov](https://codecov.io/gh/nttcslab-sp/kaldiio/branch/master/graph/badge.svg)](https://codecov.io/gh/nttcslab-sp/kaldiio)

A pure python module for reading and writing kaldi ark files

- [Introduction](#introduction)
    - [What is this? What are `ark` and `scp`?](#what-is-this-what-are-ark-and-scp)
    - [Features](#features)
    - [Similar projects](#similar-projects)
- [Install](#install)
- [Usage](#usage)
    - [ReadHelper](#readhelper)
    - [WriteHelper](#writehelper)
- [More low level API](#more-low-level-api)

## Introduction
### What are `ark` and `scp`?
`kaldiiio` is an IO utility  implemented in pure Python language for several file formats used in [kaldi](https://github.com/kaldi-asr/kaldi), which are named as`ark` and `scp`.  `ark` and `scp` are used in  in order to archive some objects defined in Kaldi, typically it is Matrix object of Kaldi.

In this section, we describe the basic concept of `ark` and `scp`. More detail about the File-IO in `Kaldi-asr`: http://kaldi-asr.org/doc/io.html


#### Basic of File IO in kaldi: Ark and copy-feats
`ark` is an archive format to save any `Kaldi objects`. This library mainly support `KaldiMatrix/KaldiVector`.
This ia an example of ark file of KaldiMatrix: [ark file](tests/arks/test.ark)

If you have `Kaldi`, you can convert it to text format as following

```bash
# copy-feats <read-specifier> <write-specifier>
copy-feats ark:test.ark ark,t:text.ark
```


`copy-feats` is designed to have high affinity with unix command line:

1. `ark` can be flushed to and from unix pipe.

        cat test.ark | copy-feats ark:- ark,t:- | less # Show the contents in the ark
    `-` indicates standard input stream or output stream. 
1. Unix command can be used as `read-specifier` and `wspecifier`

        copy-feats ark:'gunzip -c some.ark.gz |' ark:some.ark

#### Scp file
`scp` is a text file such as,

```
uttid1 /some/where/feats.ark:123
uttid2 /some/where/feats.ark:156
uttid3 /some/where/feats.ark:245
```
The first column, `uttid1`, indicates the utterance id and the second, `/some/where/feats.ark:123`, is the file path of matrix/vector of kaldi formats.  The number after colon is a starting addressof the object of the file.

`scp` looks very simple format, but has several powerful features.

1. Mutual conversion between`ark` and `scp`

        copy-feats scp:foo.scp ark:foo.ark  # scp -> ark
        copy-feats ark:foo.ark ark,scp:bar.ark,bar.scp  # ark -> ark,scp

1. Unix command can be used insead of direct file path

    For example, the following file is equivalent to the first scp.
    
        uttid1 cat /some/where/feats.ark:123 |
        uttid2 cat /some/where/feats.ark:156 |
        uttid3 cat /some/where/feats.ark:245 |

#### wav.scp
`wav.scp` is a `scp` to describe wave file paths.

```
uttid1 /some/path/a.wav
uttid2 /some/path/b.wav
uttid3 /some/path/c.wav
``` 

`wav.scp` is also can be embeded unix command as normal scp file. This is often used for converting file format in kaldi recipes.

```
uttid1 sph2pipe -f wav /some/path/a.wv1 | 
uttid2 sph2pipe -f wav /some/path/b.wv1 |
uttid3 sph2pipe -f wav /some/path/c.wv1 |
``` 

### Features
Kaldiio supports:

- Read/Write for archive formats: ark, scp
  - Binary/Text - Float/Double Matrix: DM, FM
  - Binary/Text - Float/Double Vector: DV, FV
  - Compressed Matrix for loading: CM, CM2, CM3
  - Compressed Matrix for writing: All compressoin_method are supported: 1,2,3,4,5,6,7
  - Binary/Text for Int-vector, typically used for `ali` files.
- Read/Write via a pipe: e.g. "ark: cat feats.ark |"
- Read wav.scp / wav.ark

The followings are **not supported**

- Write in existing scp file
- NNet2/NNet3 egs
- Lattice file

### Similar projects
- Python-C++ binding 
   - https://github.com/pykaldi/pykaldi
      - Looks great. I recommend pykaldi if you aren't particular about pure python.
   - https://github.com/janchorowski/kaldi-python/
      - Maybe not enough maintained now.
   - https://github.com/t13m/kaldi-readers-for-tensorflow
      - Ark reader for tensorflow
- Pure Python
   - https://github.com/vesis84/kaldi-io-for-python
      - `kaldiio` is based on this module, but `kaldiio` supports more features than it.
   - https://github.com/funcwj/kaldi-python-io
      - Python>=3.6. `nnet3-egs`is also supported.

## Install 

```bash
pip install kaldiio
```

## Usage
`kaldiio` doesn't distinguish the API for each kaldi-objects, i.e. 
`Kaldi-Matrix`, `Kaldi-Vector`, not depending on whether it is binary or text, or compressed or not, 
can be handled by the same API.

### ReadHelper
`ReadHelper` supports sequential accessing for `scp` or `ark`. If you need to access randomly, then use `kaldiio.load_scp`.


- Read matrix-scp

```python
from kaldiio import ReadHelper
with ReadHelper('scp:file.scp') as reader:
    for key, numpy_array in reader:
        ...
```


- Read gziped ark

```python
from kaldiio import ReadHelper
with ReadHelper('ark: gunzip -c file.ark.gz |') as reader:
    for key, numpy_array in reader:
        ...
        
# Ali file
with ReadHelper('ark: gunzip -c exp/tri3_ali/ali.*.gz |') as reader:
    for key, numpy_array in reader:
        ...
```


- Read wav.scp

```python
from kaldiio import ReadHelper
with ReadHelper('scp:wav.scp') as reader:
    for key, (rate, numpy_array) in reader:
        ...
```

　　　　- v2.11.0: Removed `wav` option. You can load `wav.scp` without any addtional argument.

- Read wav.scp with segments

```python
from kaldiio import ReadHelper
with ReadHelper('scp:wav.scp', segments='segments') as reader
    for key, (rate, numpy_array) in reader:
        ...
```

- Read from stdin

```python
from kaldiio import ReadHelper
with ReadHelper('ark:-') as reader:
    for key, numpy_array in reader:
        ...
```

### WriteHelper
- Write matrices in a ark with scp

```python
import numpy
from kaldiio import WriteHelper
with WriteHelper('ark,scp:file.ark,file.scp') as writer:
    for i in range(10):
        writer(str(i), numpy.random.randn(10, 10))
        # The following is equivalent
        # writer[str(i)] = numpy.random.randn(10, 10)
```

- Write in compressed matrix

```python
import numpy
from kaldiio import WriteHelper
with WriteHelper('ark:file.ark', compression_method=2) as writer:
    for i in range(10):
        writer(str(i), numpy.random.randn(10, 10))
```

- Write matrices in text

```python
import numpy
from kaldiio import WriteHelper
with WriteHelper('ark,t:file.ark') as writer:
    for i in range(10):
        writer(str(i), numpy.random.randn(10, 10))
```

- Write in gziped ark

```python
import numpy
from kaldiio import WriteHelper
with WriteHelper('ark:| gzip -c > file.ark.gz') as writer:
    for i in range(10):
        writer(str(i), numpy.random.randn(10, 10))
```
- Write matrice to stdout

```python
import numpy
from kaldiio import WriteHelper
with WriteHelper('ark:-') as writer:
    for i in range(10):
        writer(str(i), numpy.random.randn(10, 10))
```

## More low level API
`WriteHelper` and `ReadHelper` are high level wrapper of the following API to support kaldi style arguments.

### load_ark

```python
import kaldiio

d = kaldiio.load_ark('a.ark')  # d is a generator object
for key, numpy_array in d:
    ...
    
# === load_ark can accepts file descriptor, too
with open('a.ark') as fd:
    for key, numpy_array in kaldiio.load_ark(fd):
        ...

# === Use with open_like_kaldi
from kaldiio import open_like_kaldi
with open_like_kaldi('gunzip -c file.ark.gz |', 'r') as f:
    for key, numpy_array in kaldiio.load_ark(fd):
        ...
```

- `load_ark` can load both matrices of ark and vectors of ark and also, it can be both text and binary.

### load_scp
`load_scp` creates "lazy dict", i.e. 
The data are loaded in memory when accessing the element.

```python
import kaldiio

d = kaldiio.load_scp('a.scp')
for key in d:
    numpy_array = d[key]

    
with open('a.scp') as fd:
    kaldiio.load_scp(fd)
    
d = kaldiio.load_scp('data/train/wav.scp', segments='data/train/segments')
for key in d:
    rate, numpy_array = d[key]
```

The object created by `load_scp` is a dict-like object, thus it has methods of `dict`.

```python
import kaldiio
d = kaldiio.load_scp('a.scp')
d.keys()
d.items()
d.values()
'uttid' in d
d.get('uttid')
```

### load_scp_sequential (from v2.13.0)

`load_scp_sequential` creates "generator" as same as `load_ark`.
If you don't need random-accessing for each elements 
and use it just to iterate for whole data, 
then this method possibly performs faster than `load_scp`.

```python
import kaldiio
d = kaldiio.load_scp_sequential('a.scp')
for key, numpy_array in d:
    ...
```

### load_wav_scp
```python
d = kaldiio.load_scp('wav.scp')
for key in d:
    rate, numpy_array = d[key]
    
# Supporting "segments"
d = kaldiio.load_scp('data/train/wav.scp', segments='data/train/segments')
for key in d:
    rate, numpy_array = d[key]
```

- v2.11.0: `load_wav_scp` is deprecated now. Use `load_scp`.

### load_mat
```python
numpy_array = kaldiio.load_mat('a.mat')
numpy_array = kaldiio.load_mat('a.ark:1134')  # Seek and load

# If the file is wav, gets Tuple[int, numpy.ndarray]
rate, numpy_array = kaldiio.load_mat('a.wav') 
```
- `load_mat` can load kaldi-matrix, kaldi-vector, and wave

### save_ark
```python

# === Create ark file from numpy
kaldiio.save_ark('b.ark', {'key': numpy_array, 'key2': numpy_array2})
# Create ark with scp _file, too
kaldiio.save_ark('b.ark', {'key': numpy_array, 'key2': numpy_array2},
                 scp='b.scp')

# === Writes arrays to sys.stdout
import sys
kaldiio.save_ark(sys.stdout, {'key': numpy_array})

# === Writes arrays for each keys
# generate a.ark
kaldiio.save_ark('a.ark', {'key': numpy_array, 'key2': numpy_array2})
# After here, a.ark is opened with 'a' (append) mode.
kaldiio.save_ark('a.ark', {'key3': numpy_array3}, append=True)


# === Use with open_like_kaldi
from kaldiio import open_like_kaldi
with open_like_kaldi('| gzip a.ark.gz', 'w') as f:
    kaldiio.save_ark(f, {'key': numpy_array})
    kaldiio.save_ark(f, {'key2': numpy_array2})
```
### save_mat
```python
# array.ndim must be 1 or 2
kaldiio.save_mat('a.mat', numpy_array)
```
- `save_mat` can save both kaldi-matrix and kaldi-vector


### open_like_kaldi

``kaldiio.open_like_kaldi`` is a useful tool if you are familiar with Kaldi. This function can performs as following,

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

For example, if there are gziped alignment file, then you can load it as:

```python
from kaldiio import open_like_kaldi, load_ark
with open_like_kaldi('gunzip -c exp/tri3_ali/ali.*.gz |', 'rb') as f:
    # Alignment format equals ark of IntVector
    g = load_ark(f)
    for k, numpy_array in g:
        ...
```

### parse_specifier

```python
from kaldiio import parse_specifier, open_like_kaldi, load_ark
rspecifier = 'ark:gunzip -c file.ark.gz |'
spec_dict = parse_specifier(rspecifier)
# spec_dict = {'ark': 'gunzip -c file.ark.gz |'}

with open_like_kaldi(spec_dict['ark'], 'rb') as fark:
    for key, numpy_array in load_ark(fark):
        ...
```
