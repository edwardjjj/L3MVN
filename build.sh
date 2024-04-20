# create habitat env

source opt/conda/etc/profile.d/conda.sh

conda create -n habitat python=3.10 cmake=3.14.0 -y
conda activate habitat

# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# setup habitat-sim
git clone --depth 1 --branch v0.2.5 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless
cd ..

# Install habitat-lab
git clone --depth 1 --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..

pip install -r requirements.txt
# install wandb
pip install wandb

export GLOG_minloglecel=2
export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quite"
export HF_HOME=/huggingface
