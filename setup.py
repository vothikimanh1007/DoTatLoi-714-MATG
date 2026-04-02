from setuptools import setup, find_packages

setup(
    name="hypersynergy",
    version="0.8.2",  # Aligned with v82 Final MATG
    author="Vo Thi Kim Anh",
    author_email="vothikimanh@tdtu.edu.vn",
    description="A Manifold-Aware Hypergraph Framework for Heterogeneous Herbal Synergy Prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vothikimanh1007/DoTatLoi-714-MATG",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "networkx>=2.6.0",
        "matplotlib-venn>=0.11.0",
    ],
    include_package_data=True,
)
