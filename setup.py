from setuptools import setup

__author__ = 'Islam Elnabarawy'

setup(
    name='sc2agents',
    version='0.0.3',
    install_requires=['sc2gym', 'pysc2', 'baselines', 'absl', 'gym'],
    url='https://github.com/islamelnabarawy/sc2agents',
    dependency_links=['https://github.com/islamelnabarawy/sc2gym/tarball/master#egg=sc2gym-0.0.2']
)
