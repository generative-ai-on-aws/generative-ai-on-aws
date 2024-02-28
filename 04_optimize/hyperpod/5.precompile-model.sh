#!/bin/bash
cd ${APPS_PATH}/neuronx-nemo-megatron/nemo/examples/nlp/language_modeling
source ${APPS_PATH}/aws_neuron_venv_pytorch/bin/activate
sbatch --cpus-per-task 1 --nodes 4 --output ${TEST_CASE_PATH}/slurm-%x-%j.out compile.slurm ./llama_7b.sh
