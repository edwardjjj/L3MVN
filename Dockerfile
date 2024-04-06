FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

RUN mkdir -p /app
WORKDIR /app
RUN apt-get clean && apt-get update && apt-get install -y --no-install-recommends build-essential git curl vim \
  ca-certificates libjpeg-dev libglm-dev libegl1-mesa-dev xorg-dev freeglut3-dev pkg-config wget zip unzip \
  && rm -rf /var/lib/apt/lists/*

# Install miniconda

RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya

ENV PATH="/opt/conda/bin:$PATH"

# create habitat env
RUN conda create -n habitat python=3.7 cmake=3.14.0
RUN conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -n habitat
RUN conda install habitat-sim-challenge-2022 headless -c conda-forge -c aihabitat -n habitat
RUN git clone --branch challenge-2022 https://github.com/facebookresearch/habitat-lab.git
RUN conda run -n habitat /bin/bash -c \
  "cd habitat-lab; pip install -r requirements.txt; python setup.py develop --all"
RUN conda run -n habitat /bin/bash -c \
  "python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'"
COPY ./requirements.txt /home
RUN conda run -n habitat /bin/bash -c \
  "pip install -r /home/requirements.txt"
