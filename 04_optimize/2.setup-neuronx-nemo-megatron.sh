#!/usr/bin/env bash
set -euxo pipefail
APPS_PATH="$1"

cd ${APPS_PATH}
source ${APPS_PATH}/aws_neuron_venv_pytorch/bin/activate 
git clone https://github.com/aws-neuron/neuronx-nemo-megatron.git
cd neuronx-nemo-megatron
pip3 install wheel
./build.sh
pip3 install ./build/*.whl
pip3 install -r requirements.txt torch==1.13.1 protobuf==3.20.3
# You also need following to run the weight conversion
pip install accelerate 
python3 -c "from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import compile_helper; \
compile_helper()"
