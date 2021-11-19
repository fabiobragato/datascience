from setuptools import find_packages, setup

setup(
    name='datascience',
    packages=find_packages(include=['dataprocessing_fb', 'datafabric_fb']),
    version='0.2.0',
    description='Tools for Data Science',
    author='Fabio Bragato',
    author_email='fabio_bragato@hotmail.com',
    license='MIT',
    install_requires=['pandas', 'numpy', 'boto3', 'pyarrow']
)