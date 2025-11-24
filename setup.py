from setuptools import setup, find_packages

setup(
    name="ews-ml",
    version="0.1.0",
    description="Studying the mechanisms of chromatin changes in Ewing Sarcoma using deep learning",
    author="Sebastian Hönig",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.10',
)
