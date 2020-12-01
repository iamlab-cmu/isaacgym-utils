"""Setup script for carbongym_utils"""

from setuptools import setup

requirements = [
    'autolab_core',
    'autolab_perception',
    'visualization',
    'simple_zmq',
    'numba',
    'numpy-quaternion',
    'gym'
]

setup(name='carbongym_utils',
        version='0.1.0',
        author='Jacky Liang',
        author_email='jackyliang@cmu.edu',
        package_dir = {'': '.'},
        packages=['carbongym_utils'],
        install_requires=requirements
        )
