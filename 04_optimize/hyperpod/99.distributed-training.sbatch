#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=4 # number of nodes to use, 4 p4d(e) = 32 A100 GPUs
#SBATCH --job-name=FSDP # name of your job
#SBATCH --exclusive # job has exclusive use of the resource, no sharing

set -ex;

###########################
###### User Variables #####
###########################

###########################
## Environment Variables ##
###########################

## Plenty of EFA level variables
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa
export NCCL_DEBUG=INFO

###########################
####### Torch Dist  #######
###########################

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=8 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname):0 \
)

export TORCHRUN=./pt_fsdp/bin/torchrun
export TRAIN_SCRIPT=./train_fsdp.py

############################
# Llama 2 Training Params ##
############################

declare -a TRAINING_ARGS=(
    --max_context_width=4096 \
    --num_key_value_heads=32 \ # 7b: 32 13b: 40 70b: 8
    --llama_intermediate_size=11008 \ # 7b: 11008 13b: 13824 70b: 28672
    --hidden_width=4096 \ # 7b: 4096 13b: 5120 70b: 8192
    --num_layers=32 \ # 7b: 32 13b: 40 70b: 80
    --num_heads=32 \ # 7b: 32 13b: 40 70b: 64
    --model_type=llama_v2 \
    --checkpoint_freq=50 \
    --validation_freq=500 \
    --checkpoint_dir=./checkpoints \
    --resume_from_checkpoint=./checkpoints
)

srun -l ${TORCHRUN} "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"
