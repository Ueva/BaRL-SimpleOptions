import setuptools

with open("README.md", "r") as f :
    long_description = f.read()

setuptools.setup(
    name = "barl_simpleoptions",
    version = "0.0.6",
    author = "Joshua Evans",
    author_email = "jbe25@bath.ac.uk",
    description = "A package which provides a simple framework for working with Options in Reinforcement Learning.",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/Ueva/BaRL-SimpleOptions",
    packages = setuptools.find_packages(exclude=("example",)),
    install_requires=["numpy", "networkx"],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English"
    ]
)