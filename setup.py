#!/usr/bin/env python
import os.path
from setuptools import setup

setup(name='kaldiio',
      version='2.7.3',
      description='Kaldi-ark loading and writing module',
      author='Naoyuki Kamo',
      author_email='kamo_naoyuki_t7@lab.ntt.co.jp',
      long_description=open(os.path.join(os.path.dirname(__file__),
                            'README.md'), 'r').read(),
      url='http://kishin-gitlab.cslab.kecl.ntt.co.jp/kamo/kaldiio',
      packages=['kaldiio'],
      install_requires=['six', 'scipy'],
      setup_requires=['pytest-runner', 'numpy'],
      tests_require=['pytest-cov', 'pytest-html', 'pytest']
      )
