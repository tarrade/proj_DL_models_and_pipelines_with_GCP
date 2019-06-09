 # Specifies base image and tag
FROM gcr.io/deeplearning-platform-release/tf-cpu:latest

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
/bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
rm Miniconda3-latest-Linux-x86_64.sh

RUN conda info -e

# defined channels for conda
RUN conda config --append channels defaults
RUN conda config --append channels conda-forge
RUN conda config --get channels

##adding the config file in the docker image
ADD environment.yml environment.yml
ADD base.yml base.yml

# checking that the file is now present 
RUN ls -la

# update conda
RUN conda update -n base conda -y

RUN conda env update --file base.yml

# add packagesin the base to be able to see the env
RUN jupyter contrib nbextension install
RUN jupyter nbextensions_configurator enable

# install extra conda packages
RUN conda env create -f environment.yml -n env_gcp_dl

RUN conda info -e

# clean all downloaded packages
RUN conda clean -a -y 

## Configuration related to the entrerpise proxy server 
# Setup the proxy server for conda
ADD .condarc /.condarc
ENV CONDARC /.condarc
# Check that the config file was created properly
RUN conda config --show

# Setup the proxy server for pip
ADD pip.conf /pip.conf
ENV PIP_CONFIG_FILE /pip.conf

# Setup the proxy server for git
ADD .gitconfig .gitconfig
# Check that the config file was created properly
RUN git config --list
# Adding the config file to read entrerise Github
RUN mkdir -p /root/.ssh/
ADD known_hosts /root/.ssh/known_hosts
ADD id_rsa /root/.ssh/id_rsa

#adding the config file in the docker image
RUN mkdir -p /root/.ssh/
ADD known_hosts /root/.ssh/known_hosts
ADD id_rsa /root/.ssh/id_rsa


RUN conda init

#RUN echo $SHE#RUN echo $0
#RUN cat /etc/shells

#RUN /bin/sh  ~/.bashrc

#RUN conda list

#RUN conda activate base
#RUN conda activate env_gcp_dl

RUN conda list