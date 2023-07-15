# Standard library imports
from pathlib import Path
import re

# Third party imports
from setuptools import setup

__version__, = re.findall("__version__: str = '(.*)'", open('uq/__init__.py').read())

# The directory containing this file
HERE = Path(__file__).resolve().parent

README = (HERE / 'README.md').read_text()

with open('requirements.txt') as f:
	install_requires = [line.strip() for line in f.readlines()]

setup(
	name='uq',
	version=__version__,
	description='Experiments on uncertainty quantification',
	long_description=README,
	long_description_content_type='text/markdown',
	author='Victor Dheur',
	classifiers=[
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
	],
	packages=['uq'],
	include_package_data=True,
	install_requires=install_requires,
)
