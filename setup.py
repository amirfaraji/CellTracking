from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="CellTracking",
    version=0.1,
    install_requires=requirements,
    author='Lisette Lockhart, Amir Hadjifaradji',
    author_email='llockhar@sfu.ca, ahadjifaradji@gmail.com',
    packages=find_packages(),
    include_packages_data=True,
)