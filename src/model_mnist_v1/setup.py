from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['scipy>=1.1.0',
                     'pillow',
                     'matplotlib',
                     'numpy',
                     'scikit-learn>=0.20.1'
                     ]

setup(
    name='model_mnist_v1',
    version='0.1',
    author = 'F. Tarrade',
    author_email = 'fabien.tarrade@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Basic Sequetial Keras Model with MNIST data'
)