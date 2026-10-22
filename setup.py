from setuptools import setup, find_packages

# read the contents of README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="opensr-utils",
    version="2.0.0",
    author="Simon Donike, Cesar Aybar, Luis Gomez Chova, Freddie Kalaitzis",
    author_email="accounts@donike.net",
    description="Utilities supporting the ESA opensr-model package for Super-Resolution of Sentinel-2 Imagery",
    url="https://opensr.eu/",
    project_urls={"Source Code": "https://github.com/ESAopenSR/opensr-utils"},
    license="MIT",
    packages=find_packages(include=["opensr_utils", "opensr_utils.*"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Keep the public API usable on the long-lived Python 3.9/3.10
    # environments used by existing ESA processing workflows. Newer CI jobs
    # exercise the current PyTorch wheels through Python 3.14.
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    install_requires=[
        "numpy",
        "einops",
        "rasterio",
        "tqdm",
        "torch>=2.0.0",
        "scikit-image",
        "torchvision",
        "pytorch-lightning>=2.0.0",
        "xarray",
        "matplotlib",
        "omegaconf",
        "requests",
        "Pillow",
    ],
    extras_require={
        "model": ["opensr-model>=1.0.0"],
        # SEN2SR and MLSTAC currently require Python >=3.10. Keeping these
        # integrations optional preserves the core package's Python 3.9 floor.
        "sen2sr": ["sen2sr>=0.8.5", "mlstac>=0.4.9"],
        # opensr-srgan supports Python >=3.12,<3.15 upstream.
        "srgan": ["opensr-srgan>=0.5.1"],
        "all-models": [
            "opensr-model>=1.0.0",
            "sen2sr>=0.8.5",
            "mlstac>=0.4.9",
            "opensr-srgan>=0.5.1",
        ],
        "cog": ["rio-cogeo"],
        # Keep coverage tooling in the development extra so local and CI
        # measurements exercise the same dependency set as the test suite.
        "dev": ["pytest", "pytest-cov", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "opensr-run=opensr_utils.cli:main",
        ],
    },
)
