#!/usr/bin/env python3

from setuptools import setup, find_packages
print(find_packages())

setup(
    name="jagerml",
    version="0.0.1",
    url='https://github.com/imocanu/jagerml',
    description='tool with a shot of machine learning optimizations',
    packages=['jagerml'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    author="imocanu",
    author_email="iulian.mocanux@gmail.com",
    license="MIT",
    python_requires=">=3.8.0",
    install_requires=['numpy'],
    platforms=["any"],
    include_package_data=True
)

