# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import re
import pickle
import statistics
import time
import warnings
from pathlib import Path

import torch
import torch.distributed as dist

# pylint: disable=import-error,no-name-in-module
from torch.distributed import checkpoint
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from model_utils.train_utils import get_logger

logger = get_logger()

def save_checkpoint(model, optimizer, scheduler, user_content, root_dir, sub_dir):
    torch.cuda.empty_cache()

    save_dir = os.path.join(root_dir, sub_dir)
    
    with FSDP.state_dict_type(
            model, 
            StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
            "scheduler": scheduler.state_dict(),
            "total_steps": user_content["total_steps"],
            "start_batch_index": user_content["start_batch_index"],
        }
        if dist.get_rank() == 0:
            logger.info("Writing checkpoint to {0}.".format(save_dir))
        checkpoint.save_state_dict(
                    state_dict=state_dict,
                    storage_writer=checkpoint.FileSystemWriter(save_dir)
                )
    dist.barrier()
    if dist.get_rank() == 0:
        logger.info("Completed checkpoint.")

def get_last_checkpoint(checkpoint_paths, model_type):
    steps = [int(re.findall(r'\d+steps', checkpoint.stem)[0].replace('steps','')) \
         for checkpoint in checkpoint_paths]
    checkpoints = sorted([(step, path) for step,path in zip(steps, checkpoint_paths)])
    return checkpoints[-1][1].as_posix()
    
def load_checkpoint(model, optimizer, scheduler, checkpoint_dir, model_type, device):
    checkpoint_paths = list(Path(checkpoint_dir).glob(f"{model_type}*"))
    if len(checkpoint_paths)==0:
        if dist.get_rank() == 0:
            logger.info("No Checkpoints Found")
        return(
            model,
            optimizer,
            scheduler,
            0,
            0,
        )
    last_checkpoint = get_last_checkpoint(checkpoint_paths, model_type)
    if dist.get_rank() == 0:
        logger.info("Loading checkpoint from %s ...", last_checkpoint)
    with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        ):
        state_dict = {
            "model": model.state_dict(),
            "scheduler": scheduler.state_dict(),
            "total_steps": 0,
            "start_batch_index": 0,
            # cannot load the optimizer state_dict together with the model state_dict
        }
        checkpoint.load_state_dict(
            state_dict=state_dict,
            storage_reader=checkpoint.FileSystemReader(last_checkpoint),
        )
        model.load_state_dict(state_dict["model"])
        scheduler.load_state_dict(state_dict["scheduler"])
        if dist.get_rank() == 0:
            logger.info("Loaded model state from disk")
            logger.info("Loading optimizer state from disk")
        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optimizer",
            storage_reader=checkpoint.FileSystemReader(last_checkpoint),
        )
        if dist.get_rank() == 0:
            logger.info("Loaded and sharded optimizer state from disk")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # UserWarning to replace all_gather_base with all_gather_into_tensor floods the logs
            flattened_osd = FSDP.optim_state_dict_to_load(
                optim_state["optimizer"], model, optimizer
            )

        if dist.get_rank() == 0:
            logger.info("Converted optimizer state dict for FSDP")
        optimizer.load_state_dict(flattened_osd)
    dist.barrier()
    if dist.get_rank() == 0:
        logger.info("Checkpoint loaded from %s.", last_checkpoint)
    return (
        model,
        optimizer,
        scheduler,
        state_dict["total_steps"],
        state_dict["start_batch_index"],
    )

