from setuptools import setup
import glob
import os.path as path
from os import listdir
import sys
import os
from hera_gpu import version
import os.path as op
import json

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(op.join('hera_gpu', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

setup_args = {
    'name': 'hera_gpu',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_gpu',
    'license': 'BSD',
    'description': 'HERA GPU acceleration.',
    'package_dir': {'hera_gpu': 'hera_gpu'},
    'packages': ['hera_gpu'],
    'include_package_data': False,
    'version': version.version,
    'install_requires': [],
    'zip_safe': False,
}


if __name__ == '__main__':
    apply(setup, (), setup_args)
