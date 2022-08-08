""" setup script for scelephant """

import os.path
from setuptools import setup, find_packages

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setup(
    name='scelephant',
    version='0.0.3',
    author="Hyunsu An",
    author_email="ahs2202@gm.gist.ac.kr",
    description="SCelephant (Single-Cell Extremely Large Data Analysis Platform)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ahs2202/scelephant",
    license="GPLv3",
    packages=find_packages( ),
    include_package_data=True,
    install_requires=[
        'biobookshelf>=0.1.38',
        'zarr>=2.11.3',
        'numcodecs>=0.9.1',
        'hdbscan>=0.8.28',
        'pgzip>=0.3.1',
        'numba>=0.55.2',
        'pynndescent>=0.5.7',
        'leidenalg>=0.8.10',
        'bitarray>=2.4.1',
        'tqdm>=4.64.0',
        'igraph>=0.9.11',
#         'pyopa>=0.8.2',
#         'numpy>=1.22.4',

    ],
#    entry_points={
#        "console_scripts": [
#            "scarab=scarab.__main__:scarab",
#        ]
#    },
)
