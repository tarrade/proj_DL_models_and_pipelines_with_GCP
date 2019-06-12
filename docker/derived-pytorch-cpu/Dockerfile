# Specify the base image and tag
FROM gcr.io/deeplearning-platform-release/pytorch-cpu:latest

# Checking that the right shell is used for in the image the build and later when the container is launched from the image
RUN echo $0
SHELL ["/bin/bash", "-c"]
ENV SHELL /bin/bash
RUN echo $0

## Creating conda env
# Adding the environment files in the docker image
ADD environment.yml environment.yml
ADD base.yml base.yml

# Update conda envs
#RUN conda install -c conda-forge nodejs
#RUN conda update -n base conda -y
#RUN conda env update --file base.yml

# add packagesin the base to be able to see the env
##RUN jupyter contrib nbextension install
##RUN jupyter nbextensions_configurator enable

# install extra conda packages
#RUN conda env create -f environment.yml -n env_gcp_dl

# Checking the creation of the nevs
#RUN conda info -e

# Clean all downloaded packages
#RUN conda clean -a -y

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
ADD config /root/.ssh/config

RUN ls -la /root/.ssh

# Checking that the files are now present as well as the env variables
RUN printenv

#RUN conda init
#RUN cat /etc/shells
#RUN source  ~/.bashrc

#RUN conda list
#RUN conda activate base
#RUN conda activate env_gcp_dl

#RUN /root/miniconda3/envs/env_gcp_dl/bin/python -m ipykernel install --name 'env_gcp_dl' --display-name "Py36_env_gcp_dl"
#RUN /root/miniconda3/bin/python -m ipykernel install --name 'base' --display-name "Py_36_base"

# Checking some basics info
#RUN ps | grep `echo $$` | awk '{ print $4 }'
#RUN exec bash
#RUN ps | grep `echo $$` | awk '{ print $4 }'
#RUN uname -a

# Update to fix issue with extension of jupyter lab
#RUN npm install core-js@^3
#RUN jupyter lab build

# create user and create home directory
#RUN useradd -m jupyter
#RUN pwd && cd /home &&  mkdir jupyter && ls -la && pwd && jupyter:users jupyter
RUN adduser --disabled-password --gecos "" jupyter
#RUN useradd -rm -d /home/jupyter -s /bin/bash -g root -G sudo -u 1000 jupyter

# Checking more info
RUN id
RUN whoami
RUN cat /etc/passwd
RUN cat /etc/group

# Cheking is /home/jupyter was created
RUN ls -la
RUN ls -la  /home

# Trying to switch to jupyter user
USER jupyter
WORKDIR /home/jupyter

# Running some checks
RUN echo $HOME

# Managed to have /home working
#WORKDIR /home/

# Trying to set an entry pooint script
#ENTRYPOINT ["entryoint.sh"]
# --> test VM 1 ENTRYPOINT ["/bin/bash", "-c", "/entryoint.sh"]