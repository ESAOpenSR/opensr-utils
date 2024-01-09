from setuptools import setup, find_packages

# read the contents of README file
with open('README.md') as f:
    long_description = f.read()

setup(
    name='opensr-utils',
    version='0.1.2',
    authors = [
      { name="Simon Donike", email="author@example.com" },
      { name="Cesar Aybar"},
      { name="Luis Gomez Chova"},
      { name="Freddie Kalaitzis"},
        ],
    description = "Utils supporting the ESA openst-model package for Super-Resolution of Senintel-2 Imagery",
    Homepage = "https://isp.uv.es/opensr/"
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
	'numpy',
	'einops',
	'rasterio'
	'tqdm',
	'torch',
	'scikit-image'],
)
