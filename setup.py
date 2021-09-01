# install:  python -m pip install .
import setuptools

setuptools.setup(name='easygan',
      version='0.3',
      author='hglad',
      author_email='hanseglad@gmail.com',
      package_dir={"easygan": "easygan"},
      packages=setuptools.find_packages(),
     )
