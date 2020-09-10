#!/usr/bin/env python
import io
import os.path
from setuptools import setup

setup(name='kaldiio',
      version='2.15.1',
      description='Kaldi-ark loading and writing module',
      author='nttcslab-sp',
      # author_email='',
      url='https://github.com/nttcslab-sp/kaldiio',
      long_description=io.open(os.path.join(os.path.dirname(__file__),
                                            'README.md'),
                               'r', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      packages=['kaldiio'],
      install_requires=['numpy'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'pytest-cov', "soundfile"],
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Multimedia :: Sound/Audio :: Analysis']
      )
