from setuptools import find_packages, setup

setup(
    name='time_series',
    packages=find_packages(),
    version='0.1.0',
    description='Great tools for time series processing!',
    author='Emiel Deprost, Jeroen Van Der Donckt, Jonas Van Der Donckt',
    license='MIT',
    setup_requires=[
        "numpy~=1.18.5",
        "pandas~=1.0.4",
        "typing~=3.7.4.1",
        "scipy~=1.4.1",
        "dill~=0.3.3",
        "pathos~=0.2.7",
    ]
)
