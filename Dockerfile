FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN mkdir -p /home/app
WORKDIR /home/app
#install dependencies in the habitat conda env
RUN apt-key del 7fa2af80 \
    && rm /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/cuda.list
RUN apt-get clean && apt-get update && apt-get install -y --no-install-recommends build-essential git curl vim \
  ca-certificates libjpeg-dev libglm-dev libegl1-mesa-dev xorg-dev freeglut3-dev pkg-config wget zip unzip \
  && rm -rf /var/lib/apt/lists/*

# RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
#     chmod +x ~/miniconda.sh &&\
#     ~/miniconda.sh -b -p /opt/conda &&\
#     rm ~/miniconda.sh &&\
#     /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
#     /opt/conda/bin/conda clean -ya

# ENV PATH="/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:\
#   /usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
# RUN mkdir /opt/cmake
# RUN chmod +x cmake-3.14.0-Linux-x86_64.sh &&\
#   ./cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
# RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
# RUN cmake --version

RUN conda create -n habitat python=3.7 cmake=3.14.0
# RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
RUN conda install habitat-sim-challenge-2022 headless -c conda-forge -c aihabitat -n habitat
RUN git clone --branch challenge-2022 https://github.com/facebookresearch/habitat-lab.git
RUN conda run -n habitat /bin/bash -c \
  "cd habitat-lab; pip install -r requirements.txt; python setup.py develop --all"
RUN conda run -n habitat /bin/bash -c \
  "python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html"
WORKDIR /home/app
COPY . /home/app
RUN conda run -n habitat /bin/bash -c \
  "pip install -r requirements.txt"
# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/conda/lib"
RUN conda install -n habitat torchvision==0.8.2 -c pytorch
RUN conda install -n habitat -c conda-forge gxx_linux-64==12.2.0

# CMD conda run -n habitat /bin/bash -c \
#   "python -c \"import torch;print(torch.cuda.is_available())\""
CMD conda run -n habitat /bin/bash -c \
  "python main_llm_vis.py --split val --eval 1 --auto_gpu_config 0 \
  -n 8 --num_eval_episodes 250 --load pretrained_models/llm_model.pt \
  --use_gtsem 0 --num_local_steps 10"
