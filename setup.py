from setuptools import setup, find_packages

# read the contents of README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='opensr-utils',
    version='0.2.4',
    author = "Simon Donike, Cesar Aybar, Luis Gomez Chova, Freddie Kalaitzis",
    author_email = "accounts@donike.net",
    description = "Utilities supporting the ESA opensr-model package for Super-Resolution of Senintel-2 Imagery",
    url = "https://isp.uv.es/opensr/",
    project_urls={'Source Code': 'https://github.com/ESAopenSR/opensr-utils'},
    license='MIT',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
	'numpy==1.23.5',
	'einops==0.6.0',
	'rasterio==1.3.6',
	'tqdm==4.64.1',
	'torch==1.13.1',
	'scikit-image==0.19.3'],
)
