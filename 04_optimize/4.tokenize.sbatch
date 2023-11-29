#!/bin/bash
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task 128
#SBATCH --nodes 1

: "${APPS_PATH:=/fsx}"
: "${MODEL_PATH:=/fsx}"
: "${DATA_PATH:=/fsx/data/books}"
source ${APPS_PATH}/aws_neuron_venv_pytorch/bin/activate
python ${APPS_PATH}/neuronx-nemo-megatron/nemo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=${DATA_PATH}/book.jsonl \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type=${MODEL_PATH}/Llama2-7b-hf \
    --dataset-impl=mmap \
    --output-prefix=${DATA_PATH}/book-tokenized \
    --append-eod \
    --need-pad-id \
    --workers=128
