# setup.py
from setuptools import setup, find_packages

setup(
    name="stats_for_mols",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "pingouin",
        "scikit-posthocs",
        "rdkit"
    ],
)