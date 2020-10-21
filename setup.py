from setuptools import setup, find_packages

setup(
    name='com-agents',
    version='0.0.1',
    description='Implementation of the paper by Nautrup et. al. 2020',
    author='Henrik Steude',
    author_email='henrik.steude@gmail.com',
    url='https://github.com/hsteude/communicating_agents',
    packages=find_packages(include=['comm_agents']),
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'loguru',
        'torch',
        'joblib',
        'tqdm',
        'glob'],
    extras_require={'plotting': ['matplotlib', 'jupyter', 'plotly', 'interact']},
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest']
)

