#!/usr/bin/env python

# from distutils.core import setup
import setuptools

setuptools.setup(name='gantools',
      version='0.0.1',
      author='hglad',
      author_email='hanseglad@gmail.com',
      package_dir={"": "src"},
      packages=setuptools.find_packages(where="src"),
     )
