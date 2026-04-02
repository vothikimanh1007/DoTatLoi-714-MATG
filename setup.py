from setuptools import setup, find_packages

setup(
    name="hypersynergy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy",
        "scikit-learn",
        "pandas",
        "networkx",
        "matplotlib",
        "seaborn",
        "matplotlib-venn"
    ],
)
