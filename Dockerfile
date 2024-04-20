FROM nvidia/cudagl:10.1-devel-ubuntu18.04
RUN mkdir -p /app
WORKDIR /app
RUN apt-get clean && apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  curl \
  vim \
  ca-certificates \
  libjpeg-dev \
  libpng-dev \
  libglfw3-dev \
  libx11-dev \
  libomp-dev \
  libglm-dev \
  libegl1-mesa-dev \
  xorg-dev \
  freeglut3-dev \
  pkg-config \
  wget \
  zip \
  unzip \
  && rm -rf /var/lib/apt/lists/*

# Install miniconda

RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
     chmod +x ~/miniconda.sh &&\
     ~/miniconda.sh -b -p /opt/conda &&\
     rm ~/miniconda.sh &&\
     /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
     /opt/conda/bin/conda clean -ya

ENV PATH="/opt/conda/bin:$PATH"

