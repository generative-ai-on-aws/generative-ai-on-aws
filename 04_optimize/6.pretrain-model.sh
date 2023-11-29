#!/bin/bash
cd ${APPS_PATH}/neuronx-nemo-megatron/nemo/examples/nlp/language_modeling
source ${APPS_PATH}/aws_neuron_venv_pytorch/bin/activate
sbatch --nodes 4 --output ${TEST_CASE_PATH}/slurm-%x-%j.out run.slurm ./llama_7b.sh
