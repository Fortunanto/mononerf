from setuptools import setup, find_packages

setup(
    name='PhysicsOneShotNeRF',
    version='0.0.1',
    author='Yiftach Edelstein',
    author_email='',
    packages=find_packages(),
    python_requires='>=3',
    install_requires=[],
    # package_data={
    #     '': ['*.yml', '*.json'],
    # },
    zip_safe=False  # accessing config files without using pkg_resources. lazy for now
)