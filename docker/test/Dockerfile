# Specify the base image and tag
FROM gcr.io/deeplearning-platform-release/pytorch-cpu:m26
# gcloud container images list-tags gcr.io/deeplearning-platform-release/pytorch-cpu
#FROM continuumio/miniconda:latest

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

# Checking that the files are now present as well as the env variables
#RUN ls -la /root/.ssh
#RUN printenv

#RUN conda init
#RUN cat /etc/shells
#RUN source  ~/.bashrc

#RUN conda list
#RUN conda activate base
#RUN conda activate env_gcp_dl

# Checking some basics info
#RUN ps | grep `echo $$` | awk '{ print $4 }'
#RUN exec bash
#RUN ps | grep `echo $$` | awk '{ print $4 }'
#RUN uname -a

# create user and create home directory
#RUN useradd -m jupyter
#RUN pwd && cd /home &&  mkdir jupyter && ls -la && pwd && jupyter:users jupyter
#RUN adduser --disabled-password --gecos "" jupyter
#RUN useradd -rm -d /home/jupyter -s /bin/bash -g root -G sudo -u 1000 jupyter

# Checking more info
#RUN id
#RUN whoami
#RUN cat /etc/passwd
#RUN cat /etc/group

#RUN more /entrypoint.sh
#RUN more /run_jupyter.sh
#RUN more /root/.jupyter/jupyter_notebook_config.py

# Cheking is /home/jupyter was created
#RUN ls -la
#RUN ls -la  /home

# Trying to switch to jupyter user
#USER jupyter
#WORKDIR /home/jupyter

# Running some checks
#RUN echo $HOME

#ADD .bashrc .bashrc
#ADD entrypoint2.sh entrypoint2.sh
#RUN cp -f .bashrc /root/.bashrc

# set the right permission
RUN chmod +x /entrypoint.sh
#RUN chmod +x /entrypoint2.sh
#RUN chmod +x /root/.bashrc
# this stuff is not working-> conda not activated
#RUN . entrypoint.sh && conda activate base

# this stuff is working -> conda is activated
#RUN __conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"  && __conda_setup
RUN conda init bash && . /root/.bashrc && conda info -e # && conda activate base
#RUN  . /root/.bashrc && conda info -e # && conda activate base
#RUN echo $?
#RUN . /root/.bashrc
#RUN more /root/.bashrc
#RUN echo $PATH
#RUN conda info -e && . /root/miniconda3/etc/profile.d/conda.sh && conda activate base

# this stuff doesn't work
ENTRYPOINT ["/entrypoint.sh", "/run_jupyter.sh"]
#ENTRYPOINT ["/bin/bash", "-c", ". /entrypoint2.sh"]
#ENTRYPOINT source /entrypoint.sh /run_jupyter.sh

# this stuff is working not working
#ENTRYPOINT ["/bin/bash"]

#CMD ["source", "/entrypoint.sh"]
#CMD ["/bin/bash", "-c", ". /entrypoint.sh"]
#CMD ["/bin/bash", "/run_jupyter.sh"]
# this stuff doesn't work
#ENV TINI_VERSION v0.18.0
#ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
#RUN chmod +x /tini
#ENTRYPOINT ["/tini", "--", "/entrypoint.sh"]
#CMD [ "/bin/bash" ]
#CMD ["source", "/entrypoint.sh"]


# Managed to have /home working
#WORKDIR /home/


#root@86bfac2f6ccc:/# source entrypoint.sh
#(base) root@86bfac2f6ccc:/# more entrypoint.sh
##!/bin/bash
#. /env.sh
#. /root/miniconda3/etc/profile.d/conda.sh
#conda activate base
#exec "$@"