#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Global Lenslide Segmentation",
    author="G.Franch, A.Jarna, M.Calovi",
    author_email="franch@fbk.eu",
    url="https://github.com/user/project",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
