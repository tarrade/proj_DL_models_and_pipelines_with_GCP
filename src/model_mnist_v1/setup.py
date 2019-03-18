from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='model_mnist_v1',
    version='0.1',
    author = 'F. Tarrade',
    author_email = 'fabien.tarrade@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Basic Sequetial Keras Model with MNIST data',
    requires=[]
)