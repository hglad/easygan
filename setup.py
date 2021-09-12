# install:  python -m pip install .
import setuptools

setuptools.setup(name='easygan',
      version='0.4',
      author='hglad',
      author_email='hanseglad@gmail.com',
      package_dir={"easygan": "easygan"},
      packages=setuptools.find_packages(),
      install_requires=['matplotlib>=3.3.4',
                        'torchvision>=0.8.2',
                        'torch>=1.7.1',
                        'numpy>=1.19.2',
                        'imageio>=2.9.0']
     )
