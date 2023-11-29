# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import os


def parse_args():  # pylint: disable=too-many-statements
    """Parse args."""
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.

    opt_grp = parser.add_argument_group(
        title="optimization", description="arguments for optimization"
    )
    opt_grp.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="batch size per dp rank",  # pylint: disable=line-too-long
    )
    opt_grp.add_argument("--val_batch_size", type=int, default=4)
    opt_grp.add_argument("--max_steps", "--max_training_steps", type=int, default=5000)
    opt_grp.add_argument("--seed", type=int, default=12345)
    opt_grp.add_argument("--same_seed", type=int, default=0)
    opt_grp.add_argument("--bf16", default=1, type=int, help="automatic mixed precision training")
    opt_grp.add_argument("--grad_clip", default=1.0, type=float, help="gradient clipping")
    opt_grp.add_argument("--weight_decay", default=0.2, type=float, help="weight decay")
    opt_grp.add_argument(
        "--beta1", default=0.9, type=float, help="beta1 parameter for Adam optimizer"
    )
    opt_grp.add_argument(
        "--beta2", default=0.95, type=float, help="beta2 parameter for Adam optimizer"
    )
    opt_grp.add_argument(
        "--activation_checkpointing",
        type=int,
        default=1,
        help="enable gradient checkpointing to reduce memory consumption",
    )
    opt_grp.add_argument(
        "--llama_intermediate_size",
        type=int,
        default=11008,
        help="intermediate_size for Llama v2, a dimension associated with MLP",
    )
    opt_grp.add_argument(
        "--num_key_value_heads",
        type=int,
        default=None,
        help="num_key_value_heads for Llama v2",
    )
    parser.add_argument(
        "--logging_freq", type=int, default=1, help="number of iterations between logging"
    )
    parser.add_argument("--tensorboard_dir", type=str, nargs="+", default=None)

    model_grp = parser.add_argument_group(
        title="model", description="arguments to describe model configuration"
    )
    model_grp.add_argument("--max_context_width", type=int, default=2048)
    model_grp.add_argument("--vocab_size", type=int, default=50432)
    model_grp.add_argument("--hidden_width", type=int, default=768)
    model_grp.add_argument("--num_layers", type=int, default=12)
    model_grp.add_argument("--num_heads", type=int, default=12)
    model_grp.add_argument("--resid_pdrop", type=float, default=0.1)
    model_grp.add_argument("--embd_pdrop", type=float, default=0.1)
    model_grp.add_argument("--attn_pdrop", type=float, default=0.1)
    model_grp.add_argument("--summary_first_pdrop", type=float, default=0.1)
    model_grp.add_argument("--initializer_range", type=float, default=0.02)
    model_grp.add_argument("--model_type", type=str, default="gpt_neox")
    model_grp.add_argument("--rotary_pct", type=float, default=0.25)
    model_grp.add_argument("--rotary_emb_base", type=int, default=10000)

    fsdp_grp = parser.add_argument_group(
        title="fsdp", description="arguments for fully sharded data parallel"
    )
    fsdp_grp.add_argument("--offload_activations", type=int, default=0)
    fsdp_grp.add_argument("--activation_loading_horizon", type=int, default=2)
    fsdp_grp.add_argument("--limit_all_gathers", default=1, type=int)
    
    # learning rate
    lr_grp = parser.add_argument_group(
        title="lr", description="arguments for learning rate schedule"
    )
    lr_grp.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate.")
    lr_grp.add_argument(
        "--lr_decay_style",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine", "exponential", "plateau"],
        help="Learning rate decay function.",
    )
    lr_grp.add_argument(
        "--lr_decay_iters",
        type=int,
        default=None,
        help="number of iterations to decay learning rate over," " If None defaults to train iters",
    )
    lr_grp.add_argument(
        "--min_lr",
        type=float,
        default=1e-05,
        help="Minumum value for learning rate. The scheduler" "clip values below this threshold.",
    )
    lr_grp.add_argument(
        "--warmup",
        type=float,
        default=0.0032,
        help="Percentage of total iterations to warmup on "
        "(.01 = 1 percent of all training iters).",
    )
    lr_grp.add_argument(
        "--plateau",
        type=float,
        default=0.0,
        help="Percentage of total iterations to keep at max if using plateau lr",
    )
    io_grp = parser.add_argument_group(title="io", description="location for input and output")
    io_grp.add_argument("--dataset_path", type=str, default="c4")
    io_grp.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    io_grp.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint folder name to load from",
    )
    io_grp.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Saves partial checkpoints (model, optimizer) to this dir.",  # pylint: disable=line-too-long
    )
    io_grp.add_argument(
        "--epochs", type=int, default=3, help="times of iterating over the training dataset"
    )
    
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=1000,
        help="number of iterations between checkpointing",
    )
    parser.add_argument(
        "--validation_freq",
        type=int,
        default=None,
        help="number of iterations to print validation loss",
    )
    parser.add_argument(
        "--validation_batches",
        type=int,
        default=10,
        help="number of batches to estimate validation loss",
    )

    return parser.parse_known_args()