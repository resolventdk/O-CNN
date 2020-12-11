FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# set working directory
WORKDIR /home

# install compile tools
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    nano git gcc g++ make cmake wget cpio build-essential

# vtk lib gl dependency
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev

# download and install miniconda
ENV CONDA_DIR $HOME/miniconda3
RUN cd /tmp && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ./miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p $CONDA_DIR

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /etc/bash.bashrc

# update
RUN conda update -n base -c defaults conda

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# setup python environment
RUN conda create -n tf-1.14.0 -c conda-forge tensorflow-gpu==1.14.0 vtk==8.1.2 yacs tqdm && \
RUN echo "conda activate tf-1.14.0" > ~/.bashrc
ENV PATH /miniconda3/envs/tf-1.14.0/bin:$PATH

# OCNN
RUN cd /home/ && git clone https://github.com/resolventdk/O-CNN.git && \
    cd O-CNN && git checkout lm_detect && \
    cd octree/external && git clone --recursive https://github.com/wang-ps/octree-ext.git && \
    cd .. && mkdir build && cd build && cmake .. -DUSE_CUDA=ON && make && \
    cd ../../tensorflow/libs && python build.py
RUN rm -rf ~/O-CNN/resolvent