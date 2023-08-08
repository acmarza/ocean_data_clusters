FROM continuumio/miniconda3

# set working directory (this is also the mount point for git repo)
WORKDIR /app

# update conda
RUN conda update -n base -c defaults conda

# create the conda environment
COPY environment.yml .
RUN conda env create -f environment.yml 

# activate the environment
RUN echo "conda activate nccluster" >> ~/.bashrc
ENV PATH /opt/conda/envs/nccluster/bin:$PATH
ENV CONDA_DEFAULT_ENV nccluster

# mount point for data 
RUN mkdir /data

# update packages
RUN apt-get update -y && apt-get upgrade -y

# install required packages
RUN apt-get install -y git openbox tigervnc-standalone-server tk

# forward vnc port
EXPOSE 5901

# start vncserver
ENTRYPOINT tigervncserver -localhost no -SecurityTypes None --I-KNOW-THIS-IS-INSECURE -geometry 720x480 && /bin/bash
