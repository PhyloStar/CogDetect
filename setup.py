#!/usr/bin/env python

"""Install CogDetect."""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='cogdetect',
    version='0.1.0',
    description='Tools for automatic cognate detection using an Infomap',
    author='Taraka Rama, Gereon Kaiping',
    author_email='taraka@fripost.org',
    url='https://github.com/PhyloStar/CogDetect',
    license="GNU GPLv3",
    classifiers=[
        'Programming Language :: Python',
    ],
    scripts=["online_pmi"],
    packages=["infomapcog"],
    requires=["lingpy", "igraph"],
    install_requires=[]
)
