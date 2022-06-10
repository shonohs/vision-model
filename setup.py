import setuptools


setuptools.setup(name='vision_model',
                 author='irisdev',
                 description="Common interfaces for vision models.",
                 version='0.0.0',
                 install_requires=['torch'],
                 packages=setuptools.find_packages(exclude=['test', 'test.*']))
