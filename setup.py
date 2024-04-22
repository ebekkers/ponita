from setuptools import setup, find_packages

setup(
    name='ponita',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/ebekkers/ponita.git',
    license='MIT',
    author='ebekkers',
    author_email='e.j.bekkers@uva.nl',
    description='Ponita: Fast, Expressive SE(n) Equivariant Networks through Weight-Sharing in Position-Orientation Space',
    python_requires=">=3.10.5",
)
