#!/usr/bin/env python3
"""
Setup script for Foundation Models for Hydro-Climate Forecasting
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hydro-foundation-models",
    version="1.0.0",
    author="Zero Water Research Team",
    author_email="your.email@institution.edu",
    description="Foundation Models for Global Hydro-Climate Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/foundation-models-hydro-climate",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx_rtd_theme>=1.0.0",
            "myst-parser>=0.17.0",
            "numpydoc>=1.4.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "bokeh>=2.4.0",
            "cartopy>=0.20.0",
        ],
        "gpu": [
            "torch>=1.10.0+cu113",
            "torchvision>=0.11.0+cu113",
            "torchaudio>=0.10.0+cu113",
        ],
    },
    entry_points={
        "console_scripts": [
            "hydro-fm-train=scripts.train_foundation_models:main",
            "hydro-fm-evaluate=scripts.evaluate_zero_shot:main",
            "hydro-fm-finetune=scripts.evaluate_few_shot:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    keywords=[
        "hydrology",
        "climate",
        "forecasting", 
        "foundation models",
        "transfer learning",
        "streamflow",
        "machine learning",
        "deep learning",
        "transformers",
        "physics-informed",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/foundation-models-hydro-climate/issues",
        "Source": "https://github.com/your-username/foundation-models-hydro-climate",
        "Documentation": "https://foundation-models-hydro-climate.readthedocs.io/",
        "Paper": "https://arxiv.org/abs/XXXX.XXXXX",
    },
)
