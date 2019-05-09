# Specifies base image and tag
FROM gcr.io/deeplearning-platform-release/tf-cpu:latest

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
/bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
rm Miniconda3-latest-Linux-x86_64.sh

RUN conda info -e

#adding the config file in the docker image
ADD environment.yml environment.yml

# checking that the file is now present
RUN ls -la

# update conda
RUN conda update -n base conda -y

# install extra conda packages
RUN conda env create -f environment.yml -n env_gcp_dl

RUN conda info -e
