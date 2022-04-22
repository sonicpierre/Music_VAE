from setuptools import find_packages, setup

setup(
    name='model_creator_bird_sing',
    packages=find_packages(),
    version='0.1.0',
    description='Autoencoder_singing',
    author='VIRGAUX Pierre',
    license='MIT',
    install_requires=['tensorflow==2.3.1','mutagen','pydub','ray', 'librosa'],
)