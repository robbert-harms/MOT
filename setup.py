#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import shutil
from setuptools import setup, find_packages, Command


def load_requirements(fname):
    is_comment = re.compile('^\s*(#|--).*').match
    with open(fname) as fo:
        return [line.strip() for line in fo if not is_comment(line) and line.strip()]

with open('README.rst', 'rt') as f:
    readme = f.read()

with open('mot/__version__.py') as f:
    version_file_contents = f.read()
    ver_dic = {}
    exec(compile(version_file_contents, "mot/__version__.py", 'exec'), ver_dic)

requirements = load_requirements('requirements.txt')
requirements_tests = load_requirements('requirements_tests.txt')


info_dict = dict(
    name='mot',
    version=ver_dic["VERSION"],
    description='Parallel parameter estimation in python and opencl.',
    long_description=readme,
    author='Robbert Harms',
    author_email='robbert.harms@maastrichtuniversity.nl',
    url='https://github.com/robbert-harms/mot',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="LGPL v3",
    zip_safe=False,
    keywords='mot',
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Development Status :: 5 - Production/Stable',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
    ],
    test_suite='tests',
    tests_require=requirements_tests
)


class PrepareDebianDist(Command):
    description = "Prepares the debian dist prior to packaging."
    user_options = []

    def initialize_options(self):
        self.cwd = None

    def finalize_options(self):
        self.cwd = os.getcwd()

    def run(self):
        deb_info_path = './dist/deb/mot-{}/debian'.format(ver_dic["VERSION"])
        with open(deb_info_path + '/rules'.format(ver_dic["VERSION"]), 'a') as f:
            f.write('\noverride_dh_auto_test:\n\techo "Skip dh_auto_test"')

        self._set_copyright_file(deb_info_path)
        self._set_description(deb_info_path)

    def _set_copyright_file(self, deb_info_path):
        shutil.copy('debian/copyright', deb_info_path + '/copyright')

        with open(deb_info_path + '/copyright', 'r') as file:
            copyright_info = file.read()

        copyright_info = copyright_info.replace('{{source}}', info_dict['url'])
        copyright_info = copyright_info.replace('{{years}}', '2016-2017')
        copyright_info = copyright_info.replace('{{author}}', info_dict['author'])
        copyright_info = copyright_info.replace('{{email}}', info_dict['author_email'])

        with open(deb_info_path + '/copyright', 'w') as file:
            file.write(copyright_info)

    def _set_description(self, deb_info_path):
        with open(deb_info_path + '/control', 'r') as file:
            control = file.readlines()
            lines = []
            for line in control:
                lines.append(line)
                if line[0:len('Description')] == 'Description':
                    break

        control = "".join(lines)
        control += " A library for parallel parameter estimation in python and opencl."

        with open(deb_info_path + '/control', 'w') as file:
            file.write(control)


info_dict.update(cmdclass={'prepare_debian_dist': PrepareDebianDist})
setup(**info_dict)
