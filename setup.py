#!/usr/bin/env python

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='estorch',
      version='1.0.5',
      description='Evolution Strategy Library',
      author='Göktuğ Karakaşlı',
      author_email='karakasligk@gmail.com',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/goktug97/estorch',
      packages = ['estorch'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      install_requires=[
          'torch',
          'numpy',
          'scipy',
          'mpi4py'
      ],
      python_requires='>=3.6',
      include_package_data=True)
