# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import math
import functools
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
import tqdm
import logging
from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy

g_gigabyte = 1024**3

def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run



def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
  
    if sampler:
        sampler.set_epoch(epoch)
    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        optimizer.zero_grad()
        output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        if rank==0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]


    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy


def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
            fsdp_loss[0] += output["loss"].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

def get_model_config(args):
    if "gpt_neox" in args.model_type:
        from transformers import GPTNeoXConfig

        model_config = GPTNeoXConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_width,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            hidden_act="gelu",
            intermediate_size=4 * args.hidden_width,
            rotary_pct=args.rotary_pct,
            rotary_emb_base=args.rotary_emb_base,
            max_position_embeddings=args.max_context_width,
            layer_norm_epsilon=1e-05,
            initializer_range=args.initializer_range,
            use_cache=False,
            parallel_attn_output=True,
        )
    elif "llama_v2" in args.model_type:
        from transformers import LlamaConfig

        model_config = LlamaConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_width,
            intermediate_size=args.llama_intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            num_key_value_heads=args.num_key_value_heads,
            hidden_act="silu",
            max_position_embeddings=args.max_context_width,
            initializer_range=args.initializer_range,
            rms_norm_eps=1e-5,
            use_cache=False,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_scaling=None,
        )
    else:
        raise NotImplementedError
    return model_config

def compute_num_params(model):
    """Get num params."""
    num_params = 0
    seen = set()
    for p in model.parameters():  # pylint: disable=invalid-name
        if p not in seen:
            seen.add(p)
            if hasattr(p, "ds_shape"):
                num_params += np.prod(p.ds_shape)
            else:
                num_params += np.prod(p.size())

    return num_params

_logger = None
def get_logger():
    global _logger
    if _logger is None:
        logging.getLogger("torch.distributed.checkpoint._dedup_tensors").setLevel(logging.ERROR)
        logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)
        _logger = logging.getLogger(__name__)
        _logger.setLevel(logging.INFO)
        _logger.handlers = []
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname).1s " "[%(filename)s:%(lineno)d] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        _logger.addHandler(ch)
        _logger.propagate = False
    return _logger

def get_transformer_layer(model_type="gpt2"):
    """Get transformer layer."""
    if model_type == "gpt2":
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        transformer_layer = GPT2Block

    elif model_type == "gpt_neox":
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

        transformer_layer = GPTNeoXLayer

    elif model_type == "bloom":
        from transformers.models.bloom.modeling_bloom import BloomBlock

        transformer_layer = BloomBlock

    elif model_type == "flash_gptneox":
        from flash_attn.modules.block import ParallelBlock

        # TODO: Add support for Block
        transformer_layer = ParallelBlock
    elif model_type == "llama_v2":
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        transformer_layer = LlamaDecoderLayer
    return transformer_layer

def get_sharding_strategy(strategy: str):
    """Get sharding strategy."""
    sharding_strategy = getattr(ShardingStrategy, strategy.upper())
    _logger.debug("Translating %s to %s.", strategy, sharding_strategy)
    return sharding_strategy


def get_backward_fetch_policy(policy: str):
    """Get backward fetch policy."""
    backward_fetch_policy = getattr(BackwardPrefetch, policy.upper())
    _logger.debug("Translating %s to %s.", policy, backward_fetch_policy)
    return backward_fetch_policy

def apply_activation_checkpoint(args, model=None):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    transformer_layer = get_transformer_layer(args.model_type)
    check_fn_gpt = lambda submodule: isinstance(
        submodule, transformer_layer
    )
    entrant_wrapper = functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=entrant_wrapper, check_fn=check_fn_gpt
    )

def get_param_groups_by_weight_decay(module):
    """Get param groups."""
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    param_ids = set()

    from torch.nn import LayerNorm

    for module_ in module.modules():
        # if isinstance(module_, FusedLayerNorm) or
        if isinstance(module_, LayerNorm):
            for p in list(
                module_._parameters.values()
            ):  # pylint: disable=invalid-name,protected-access
                if p is not None and id(p) not in param_ids:
                    no_weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
        else:
            for n, p in list(
                module_._parameters.items()
            ):  # pylint: disable=invalid-name,protected-access
                if p is not None and n != "bias" and id(p) not in param_ids:
                    weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
            for n, p in list(
                module_._parameters.items()
            ):  # pylint: disable=invalid-name,protected-access
                if p is not None and n == "bias" and id(p) not in param_ids:
                    no_weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
    return weight_decay_params, no_weight_decay_params

class AnnealingLR:  # pylint: disable=too-many-instance-attributes
    """Anneals the learning rate."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        optimizer,
        start_lr,
        warmup_iter,
        plateau_iter,
        total_iters,
        decay_style,
        last_iter,
        min_lr=0.0,
        use_checkpoint_lr_scheduler=True,
        override_lr_scheduler=False,
    ):

        # Class values.
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.warmup_iter = warmup_iter
        self.plateau_iter = plateau_iter
        self.num_iters = last_iter
        self.end_iter = total_iters
        assert self.end_iter > 0
        self.decay_style = decay_style
        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler
        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, (
                "both override and " "use-checkpoint are set."
            )
        # Set the learning rate
        self.step(self.num_iters)
        self.rank = dist.get_rank()

    def get_lr(self):
        """Learning rate decay functions from:
        https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        num_iters_ = min(self.num_iters, self.end_iter - self.warmup_iter)
        # Warmup.
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return float(self.start_lr) * num_iters_ / self.warmup_iter

        num_iters_ = num_iters_ - self.warmup_iter
        if self.decay_style == "linear":
            lr = self.start_lr * (self.end_iter - num_iters_) / self.end_iter
        elif self.decay_style == "plateau":
            if self.num_iters <= self.plateau_iter:
                lr = self.start_lr
            else:
                lr = (
                    self.start_lr
                    * (self.end_iter - self.num_iters)
                    / (self.end_iter - self.plateau_iter)
                )
        elif self.decay_style == "cosine":
            lr = self.start_lr / 2.0 * (math.cos(math.pi * num_iters_ / self.end_iter) + 1)
        elif self.decay_style == "exponential":
            # exp(-0.693) = 1/2
            lr = self.start_lr * math.exp(-0.693 * num_iters_ / self.end_iter)
        else:
            lr = self.start_lr
        return max(lr, self.min_lr)

    def step(self, step_num=None):
        """Set lr for all parameters groups."""
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def state_dict(self):
        """State dict."""
        state_dict = {
            "start_lr": self.start_lr,
            "warmup_iter": self.warmup_iter,
            "num_iters": self.num_iters,
            "decay_style": self.decay_style,
            "end_iter": self.end_iter,
            "min_lr": self.min_lr,
        }
        return state_dict

    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_lr_scheduler:
            if self.rank == 0:
                _logger.info(f"Overriding {name} value to {cls_value}")
            return cls_value

        if not self.use_checkpoint_lr_scheduler:
            assert (
                cls_value == sd_value
            ), f"AnnealingLR: class input value and checkpoint values for {name} do not match"
        if self.rank == 0:
            _logger.info(f" > using checkpoint value {sd_value} for {name}")
        return sd_value

    def load_state_dict(self, sd):
        """Load state dict."""
        self.start_lr = self._check_and_set(self.start_lr, sd["start_lr"], "learning rate")
        self.min_lr = self._check_and_set(self.min_lr, sd["min_lr"], "minimum learning rate")
        self.warmup_iter = self._check_and_set(
            self.warmup_iter, sd["warmup_iter"], "warmup iterations"
        )
        self.end_iter = self._check_and_set(
            self.end_iter, sd["end_iter"], "total number of iterations"
        )
        self.decay_style = self._check_and_set(self.decay_style, sd["decay_style"], "decay style")

        self.num_iters = sd["num_iters"]
        self.step(self.num_iters)

def get_learning_rate_scheduler(optimizer, args):
    """Get learning rate scheduler."""
    use_checkpoint_lr_scheduler = args.resume_from_checkpoint is not None

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.max_steps
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = args.warmup * num_iters
    plateau_iter = warmup_iter + args.plateau * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=warmup_iter,
        plateau_iter=plateau_iter,
        total_iters=num_iters,
        decay_style=args.lr_decay_style,
        last_iter=init_step,
        min_lr=args.min_lr,
        use_checkpoint_lr_scheduler=use_checkpoint_lr_scheduler,
        override_lr_scheduler=False,
    )

    return lr_scheduler