#!/bin/bash
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task 96
#SBATCH --nodes 1

: "${APPS_PATH:=/fsx}"
: "${MODEL_PATH:=/fsx}"
: "${DATA_PATH:=/fsx/data/books}"

source ${APPS_PATH}/aws_neuron_venv_pytorch/bin/activate
python ${APPS_PATH}/aws_neuron_venv_pytorch/lib/python3.8/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir ${MODEL_PATH}/Llama2-meta --model_size 7B --output_dir ${MODEL_PATH}/Llama2-7b-hf
