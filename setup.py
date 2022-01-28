from glob import glob
import os
from os.path import join as pjoin
from setuptools import setup, find_packages

from jupyter_packaging import (
    create_cmdclass,
    install_npm,
    ensure_targets,
    combine_commands,
    get_version,
    skip_if_exists
)

HERE = os.path.dirname(os.path.abspath(__file__))

name = "toywidget"

jstargets = [
    pjoin(HERE, "databutler_widgets", name, "nbextension", "index.js"),
    pjoin(HERE, "databutler_widgets", name, "labextension", "package.json"),
]

package_data_spec = {
    name: [
        "nbextension/**js*",
        "labextension/**"
    ]
}

data_files_spec = [
    ('share/jupyter/nbextensions/toywidget', 'databutler_widgets/toywidget/nbextension', '**'),
    ('share/jupyter/labextensions/toywidget', 'databutler_widgets/toywidget/labextension', '**'),
    ('share/jupyter/labextensions/toywidget', 'databutler_widgets_src/toywidget', 'install.json'),
    ('etc/jupyter/nbconfig/notebook.d', 'databutler_widgets_src/toywidget', 'toywidget.json'),
]

cmdclass = create_cmdclass('jsdeps', package_data_spec=package_data_spec,
                           data_files_spec=data_files_spec)
npm_install = combine_commands(
    install_npm(pjoin(HERE, "databutler_widget_srcs", "toywidget"), build_cmd='build:prod'),
    ensure_targets(jstargets),
)
cmdclass['jsdeps'] = skip_if_exists(jstargets, npm_install)

setup_args = dict(
    name='DataButler',
    version='0.0.0',
    packages=find_packages(),
    cmdclass=cmdclass,
    url='https://github.com/rbavishi/DataButler',
    license='BSD-2-Clause',
    author='Rohan Bavishi',
    author_email='rbavishi@cs.berkeley.edu',
    description='A butler for your data'
)

if __name__ == '__main__':
    setup(**setup_args)
