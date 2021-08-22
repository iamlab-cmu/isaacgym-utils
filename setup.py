"""Setup script for isaacgym_utils"""

from setuptools import setup

requirements = [
    'autolab_core',
    'autolab_perception',
    'visualization',
    'simple_zmq',
    'triangle',
    'numba',
    'numpy-quaternion',
    'gym',
    'stable_baselines3',
]

setup(name='isaacgym_utils',
        version='0.1.0',
        author='Jacky Liang',
        author_email='jackyliang@cmu.edu',
        package_dir = {'': '.'},
        packages=['isaacgym_utils'],
        install_requires=requirements
        )
