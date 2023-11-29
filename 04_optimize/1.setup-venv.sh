#!/usr/bin/env bash
set -euxo pipefail
APPS_PATH="$1"

cd ${APPS_PATH}
# Install Python venv 
sudo apt-get install -y python3.8-venv g++ 

# Create Python venv
python3.8 -m venv aws_neuron_venv_pytorch 

# Activate Python venv 
source ${APPS_PATH}/aws_neuron_venv_pytorch/bin/activate 
python -m pip install -U pip 

# Install Jupyter notebook kernel
pip install ipykernel 
python3.8 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
pip install jupyter notebook
pip install environment_kernels

# Set pip repository pointing to the Neuron repository 
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install wget, awscli 
python -m pip install wget 
python -m pip install awscli 

# Install Neuron Compiler and Framework
python -m pip install neuronx-cc==2.* torch-neuronx torchvision