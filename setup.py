# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="textless",
    version="0.1.0",
    url="https://github.com/facebookresearch/textlesslib",
    author="Textless NLP team at Facebook AI Research",
    author_email="kharitonov@fb.com",
    description="Tools for Textless NLP Research",
    packages=find_packages(),
    install_requires=requirements,
)
