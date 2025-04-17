# Created by Javad Komijani (2025)

"""This is the setup script for `lattice_ml`."""

from setuptools import setup, find_packages


def readme():
    """Reads and returns the contents of the README.md file."""
    with open('README.md', encoding='utf-8') as f:
        return f.read()


setup(
    name='lattice_ml',
    version='1.0.0',
    description='A PyTorch-based library for lattice field theory',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Javad Komijani',
    author_email='jkomijani@gmail.com',
    url='http://github.com/jkomijani/lattice_ml',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    zip_safe=False,
)
