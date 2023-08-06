FROM continuumio/miniconda3

# set working directory
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

# copy over core scripts
COPY nccluster /app/nccluster
COPY examples /app/examples

# mount points for config files and netCDF data
RUN mkdir configs
RUN mkdir data

# update packages
RUN apt-get update -y && apt-get upgrade -y

# install graphics packages
RUN apt-get install -y tigervnc-standalone-server tk

# forward vnc port
EXPOSE 5901

# start vncserver
ENTRYPOINT tigervncserver -localhost no -SecurityTypes None --I-KNOW-THIS-IS-INSECURE && /bin/bash
