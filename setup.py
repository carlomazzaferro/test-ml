#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from pathlib import Path

EXTENSIONS = {
    'tf',
    'tf_gpu',
    'sklearn'
}


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()


def strip_comments(l):
    return l.split('#', 1)[0].strip()


def _pip_requirement(req, *root):
    if req.startswith('-r '):
        _, path = req.split()
        return reqs(*root, *path.split('/'))
    return [req]


def _reqs(*f):
    path = (Path.cwd() / 'requirements').joinpath(*f)
    with path.open() as fh:
        reqs = [strip_comments(l) for l in fh.readlines()]
        return [_pip_requirement(r, *f[:-1]) for r in reqs if r]


def reqs(*f):
    return [req for subreq in _reqs(*f) for req in subreq]


def extras(*p):
    """Parse requirement in the requirements/extras/ directory."""
    return reqs('extras', *p)


def extras_require():
    """Get map of all extra requirements."""
    return {x: extras(x + '.txt') for x in EXTENSIONS}


install_requires = reqs('base.txt')
test_requires = reqs('test.txt') + install_requires

setup(
    author="Carlo Mazzaferro",
    author_email='carlo.mazzaferro@u.group',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Opinionated scaffolding library for machine learning projects",
    entry_points={
        'console_scripts': [
            'testml=testml.main:cli',
        ],
    },
    install_requires=install_requires,
    extras_require=extras_require(),
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='testing',
    name='test-ml',
    packages=find_packages(include=['testml']),
    setup_requires=install_requires,
    test_suite='tests',
    tests_require=test_requires,
    url='https://github.com/carlomazzaferro/test-ml',
    version='0.1.0',
    zip_safe=False,
)
