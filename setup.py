"""Setup script for isaacgym_utils"""

from setuptools import setup

# Optional dependency groups.
extras = {
    "ray": ["ray"],
    "rl": ["stable_baselines3"],
}

extras["all"] = list(
    set([item for group in extras.values() for item in group])
)

requirements = [
    'autolab_core',
    'autolab_perception',
    'visualization',
    'simple_zmq',
    'triangle',
    'numba',
    'numpy-quaternion',
    'gym',
]

setup(name='isaacgym_utils',
        version='0.1.0',
        author='Jacky Liang',
        author_email='jackyliang@cmu.edu',
        package_dir = {'': '.'},
        packages=['isaacgym_utils'],
        install_requires=requirements,
        extras_require=extras,
        )
