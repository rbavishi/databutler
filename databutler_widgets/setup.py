#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from glob import glob
import os
from os.path import join as pjoin
from setuptools import setup, find_packages

from jupyter_packaging import (
    create_cmdclass,
    install_npm,
    ensure_targets,
    combine_commands,
    skip_if_exists
)

HERE = os.path.dirname(os.path.abspath(__file__))

# The name of the project
name = 'databutler_widgets'

# Representative files that should exist after a successful build
jstargets = [
    pjoin(HERE, name, 'nbextension', 'index.js'),
    pjoin(HERE, name, 'labextension', 'package.json'),
]

package_data_spec = {
    name: [
        'nbextension/**js*',
        'labextension/**'
    ]
}

data_files_spec = [
    ('share/jupyter/nbextensions/databutler_widgets', 'databutler_widgets/nbextension', '**'),
    ('share/jupyter/labextensions/databutler_widgets', 'databutler_widgets/labextension', '**'),
    ('share/jupyter/labextensions/databutler_widgets', '.', 'install.json'),
    ('etc/jupyter/nbconfig/notebook.d', '.', 'databutler_widgets.json'),
]

cmdclass = create_cmdclass('jsdeps', package_data_spec=package_data_spec,
                           data_files_spec=data_files_spec)
npm_install = combine_commands(
    install_npm(HERE, build_cmd='build:prod'),
    ensure_targets(jstargets),
)
cmdclass['jsdeps'] = npm_install

setup_args = dict(
    name=name,
    description='Jupyter widgets for DataButler',
    version="0.0.0",
    scripts=glob(pjoin('scripts', '*')),
    cmdclass=cmdclass,
    packages=find_packages(),
    author='Rohan Bavishi',
    author_email='rbavishi@cs.berkeley.edu',
    url='https://github.com/rbavishi/databutler',
    license='BSD',
    platforms="Linux, Mac OS X, Windows",
    keywords=['Jupyter', 'Widgets', 'IPython'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Framework :: Jupyter',
    ],
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        'ipywidgets>=7.0.0',
    ],
)

if __name__ == '__main__':
    setup(**setup_args)
