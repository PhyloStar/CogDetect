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
    author='Taraka Rama',
    author_email='taraka@fripost.org',
    url='https://github.com/PhyloStar/CogDetect',
    license="???",
    classifiers=[
        'Programming Language :: Python',
    ],
    scripts=['online_pmi'],
    py_modules=['ipa2asjp'],
    requires=['lingpy'],
    install_requires=[]
)
