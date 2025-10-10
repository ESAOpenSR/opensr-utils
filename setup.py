from setuptools import setup, find_packages

# read the contents of README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='opensr-utils',
    version='1.1.2',
    author = "Simon Donike, Cesar Aybar, Luis Gomez Chova, Freddie Kalaitzis",
    author_email = "accounts@donike.net",
    description = "Utilities supporting the ESA opensr-model package for Super-Resolution of Senintel-2 Imagery",
    url = "https://opensr.eu/",
    project_urls={'Source Code': 'https://github.com/ESAopenSR/opensr-utils'},
    license='MIT',
    packages=find_packages(include=["opensr_utils", "opensr_utils.*"]),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'einops',
        'rasterio',
        'tqdm',
        'torch',
        'scikit-image',
        'torchvision',
        'pytorch-lightning',
        'xarray',
        'matplotlib',
        'omegaconf',
        'requests',
    ],
        entry_points={
                "console_scripts": [
                    "opensr-run=opensr_utils.cli:main",
                ],
            },
)
