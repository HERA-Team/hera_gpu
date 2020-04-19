import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="hera_gpu",
	version="2019.06.1",
	author="Jackson Sipple",
	author_email="jsipple@berkeley.edu",
	description="Drop-in replacement functions that run on NVIDIA GPUs, parallelized for speed",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/HERA-Team/hera_gpu",
	packages=setuptools.find_packages(),
	install_requires=[
		'pycuda',
		'numpy',
		'scipy',
		'scikit-cuda',
	],
	dependency_links=['https://github.com/HERA-Team/hera_sim'],
	classifiers=(
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent",
	)
)
