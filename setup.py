#!/usr/bin/env python
import os
from setuptools import setup

setup(name='kaldiio',
      version='2.12.0',
      description='Kaldi-ark loading and writing module',
      author='Naoyuki Kamo',
      author_email='kamo_naoyuki_t7@lab.ntt.co.jp',
      url='https://github.com/nttcslab-sp/kaldiio',
      long_description=os.open(os.path.join(os.path.dirname(__file__),
                                            'README.md'),
                               'r', encoding='utf-8').read(),
      packages=['kaldiio'],
      install_requires=['six', 'scipy'],
      setup_requires=['pytest-runner', 'numpy'],
      tests_require=['pytest', 'pytest-cov']
      )
