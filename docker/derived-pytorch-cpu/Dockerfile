# Specify the base image and tag
FROM gcr.io/deeplearning-platform-release/pytorch-cpu:m26

# Checking the type of machine
RUN cat /etc/os-release
RUN lsb_release -a
RUN uname -r

# Checking that the right shell is used in the image the build and later when the container is launched from the image
RUN echo $0
# active in the shell
SHELL ["/bin/bash", "-c"]
# active in the terminla later
ENV SHELL /bin/bash
RUN echo $0

## Creating conda env
# Adding the environment files in the docker image
ADD base.yml base.yml
ADD environment.yml environment.yml
ADD environment_2_0.yml environment_2_0.yml
ADD environment_2_0_nightly.yml environment_2_0_nightly.yml

# Update Conda manager
RUN conda update -n base conda -y

# Update conda envs
RUN conda env update --file base.yml

# Install env
RUN conda env create -f environment.yml
RUN conda env create -f environment_2_0.yml
RUN conda env create -f environment_2_0_nightly.yml

# Checking the creation of the nevs
RUN conda info -e

# Clean all downloaded packages
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
ADD config /root/.ssh/config

# Checking that the files are now present as well as the env variables
#RUN ls -la /root/.ssh
#RUN printenv

#RUN conda init
#RUN cat /etc/shells
#RUN source  ~/.bashrc

#RUN conda list
#RUN conda activate base
#RUN conda activate env_gcp_dl

# Create Kernel
RUN /root/miniconda3/bin/python -m ipykernel install --name 'base' --display-name "env_base_py36"
RUN /root/miniconda3/envs/env_gcp_dl/bin/python -m ipykernel install --name 'env_gcp_dl' --display-name "env_gcp_dl_py36"
RUN /root/miniconda3/envs/env_gcp_dl_2_0_beta/bin/python -m ipykernel install --name 'env_gcp_dl_2_0_beta' --display-name "env_2_0_beta_py36"
RUN /root/miniconda3/envs/env_gcp_dl_2_0_nightly/bin/python -m ipykernel install --name 'env_gcp_dl_2_0_nightly' --display-name "env_2_0_nightly_py36"

# Checking some basics info
#RUN ps | grep `echo $$` | awk '{ print $4 }'
#RUN exec bash
#RUN ps | grep `echo $$` | awk '{ print $4 }'
#RUN uname -a

# Update to fix issue with extension of jupyter lab
RUN jupyter labextension list
RUN npm install --save core-js@^3
RUN jupyter labextension install @jupyterlab/latex
RUN jupyter labextension install @mflevine/jupyterlab_html
RUN jupyter labextension install jupyterlab-drawio
RUN jupyter labextension install @jupyterlab/plotly-extension
RUN jupyter labextension install jupyterlab_bokeh
RUN jupyter labextension install @jupyterlab/toc
RUN jupyter labextension install @jupyterlab/github
RUN jupyter contrib nbextension install
RUN jupyter nbextensions_configurator enable
RUN jupyter lab build

# create user and create home directory
#RUN useradd -m jupyter
#RUN pwd && cd /home &&  mkdir jupyter && ls -la && pwd && jupyter:users jupyter
#RUN adduser --disabled-password --gecos "" jupyter
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
#USER jupyter
#WORKDIR /home/jupyter

# Running some checks
RUN echo $HOME

# set the right permission
RUN chmod +x /entrypoint.sh

# this stuff is working -> conda is activated
#RUN . entrypoint.sh && conda init bash && source /root/.bashrc && conda info -e && conda activate base
RUN conda init bash && . /root/.bashrc && conda info -e # && conda activate base
# this stuff doesn't work
#ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
#ENTRYPOINT source /entrypoint.sh

# this stuff is working not working
#ENTRYPOINT ["/bin/bash"]

# this suff is working ?
#CMD ["source", "/entrypoint.sh"]
ENTRYPOINT ["/entrypoint.sh", "/run_jupyter.sh"]
# this stuff doesn't work
#ENV TINI_VERSION v0.18.0
#ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
#RUN chmod +x /tini
#ENTRYPOINT ["/tini", "--", "/entrypoint.sh"]
#CMD [ "/bin/bash" ]
#CMD ["source", "/entrypoint.sh"]


# Managed to have /home working
WORKDIR /home/