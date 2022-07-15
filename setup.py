"""Setup script for Scarab """

import os.path
from setuptools import setup, find_packages

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setup(
    name='scelephant',
    version='0.0.0',
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
        'biobookshelf>=0.1.36',
        'zarr>=2.11.3',
        'numcodecs>=0.9.1',
        'hdbscan>=0.8.28',
#         'pyopa>=0.8.2',
#         'numpy>=1.22.4',
#         'numba>=0.55.2',
    ],
#    entry_points={
#        "console_scripts": [
#            "scarab=scarab.__main__:scarab",
#        ]
#    },
)
