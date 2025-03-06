#!/usr/bin/env python
import io
import os.path
from setuptools import setup

setup(
    name="kaldiio",
    version="2.18.1",
    description="Kaldi-ark loading and writing module",
    author="nttcslab-sp",
    # author_email='',
    url="https://github.com/nttcslab-sp/kaldiio",
    long_description=io.open(
        os.path.join(os.path.dirname(__file__), "README.md"), "r", encoding="utf-8"
    ).read(),
    long_description_content_type="text/markdown",
    packages=["kaldiio"],
    install_requires=["numpy"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov", "soundfile"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
)
