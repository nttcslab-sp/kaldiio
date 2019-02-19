#!/usr/bin/env python
import io
import os.path
from setuptools import setup

setup(name='kaldiio',
      version='2.13.2',
      description='Kaldi-ark loading and writing module',
      author='nttcslab-sp',
      # author_email='',
      url='https://github.com/nttcslab-sp/kaldiio',
      long_description=io.open(os.path.join(os.path.dirname(__file__),
                                            'README.md'),
                               'r', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      packages=['kaldiio'],
      install_requires=['six', 'scipy'],
      setup_requires=['pytest-runner', 'numpy'],
      tests_require=['pytest', 'pytest-cov']
      )
