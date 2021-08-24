# from distutils.core import setup
import setuptools

setuptools.setup(name='gantools',
      version='0.0.2',
      author='hglad',
      author_email='hanseglad@gmail.com',
      package_dir={"gantools": "gantools"},
      # packages=setuptools.find_packages(where="src"),
      packages=setuptools.find_packages(),
     )
