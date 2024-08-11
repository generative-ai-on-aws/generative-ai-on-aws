"""GPT 또는 BLOOM 모델 학습"""

# pylint: disable=protected-access,too-many-lines

import argparse
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from modelling_RW import RWForCausalLM
import model_config as model_config_lib
import numpy as np
import smdistributed.modelparallel
import smdistributed.modelparallel.torch as smp
import torch
import torch.utils.data
import transformers
from data_pipeline import create_pretraining_dataloader  # pylint: disable=wrong-import-order
from learning_rates import AnnealingLR  # pylint: disable=wrong-import-order
from memory_tracker import memory_status, memory_status_cpu  # pylint: disable=wrong-import-order
from sdp_utils import build_param_id_to_buffer, build_param_id_to_offset, log_param_norms
from smdistributed.modelparallel.torch.nn import FusedLayerNorm  # pylint: disable=import-error
from torch import optim
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig, set_seed
from transformers.trainer_utils import is_main_process

# pylint: enable=import-error


logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def get_learning_rate_scheduler(optimizer, args):
    """학습률 스케줄러 가져오기"""
    # 선형 학습률 스케줄러 추가
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
        use_checkpoint_lr_scheduler=args.load_partial or args.load_full,
        override_lr_scheduler=False,
    )

    return lr_scheduler


def get_param_groups_by_weight_decay(module):
    """매개변수 그룹 가져오기"""
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    param_ids = set()
    for module_ in module.modules():
        if isinstance(module_, FusedLayerNorm):
            for p in list(module_._parameters.values()):  # pylint: disable=invalid-name
                if p is not None and id(p) not in param_ids:
                    no_weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
        else:
            for n, p in list(  # pylint: disable=invalid-name
                module_._parameters.items()  # pylint: disable=protected-access
            ):
                if p is not None and n != "bias" and id(p) not in param_ids:
                    weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
            for n, p in list(  # pylint: disable=invalid-name
                module_._parameters.items()  # pylint: disable=protected-access
            ):
                if p is not None and n == "bias" and id(p) not in param_ids:
                    no_weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
    if not no_weight_decay_params["params"]:
        return [weight_decay_params]
    return weight_decay_params, no_weight_decay_params


# smdistributed: smp.step 정의. 외부에서 필요한 텐서를 반환합니다.
@smp.step
def train_step(model, input_ids, attention_mask, args):
    """Train step."""
    if args.logits_output:
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = output["loss"]
    else:
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)["loss"]
    model.backward(loss)
    if args.logits_output:
        return output

    return loss


# smdistributed: smp.step 정의. 외부에서 필요한 텐서를 반환합니다.
@smp.step
def test_step(model, input_ids, attention_mask):
    """학습 단계"""
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)["loss"]
    return loss


def eval_model(model, dataloader, num_batches, use_bert_data):
    """모델 평가"""
    model = model.eval()
    n_batches = 0
    loss = 0.0

    with torch.no_grad():
        for batch_idx, input_data in enumerate(dataloader):
            if use_bert_data:
                input_ids, _, attention_mask, _, _ = input_data
            else:
                input_ids, attention_mask = input_data
            if batch_idx >= num_batches:
                break

            loss += test_step(model, input_ids, attention_mask).reduce_mean()
            n_batches += 1

    if n_batches > 0:
        torch.distributed.all_reduce(loss, group=smp.get_dp_process_group())
        loss /= smp.dp_size()
        loss /= n_batches
        loss = loss.item()
        ppl = math.exp(loss)
    else:
        loss = -1.0
        ppl = -1.0

    return loss, ppl


def train(  # pylint: disable=too-many-arguments,too-many-branches,too-many-locals,too-many-statements
    model,
    optimizer,
    lr_scheduler,
    model_config,
    start_train_path_index,
    start_batch_index,
    num_params,
    total_steps,
    args,
    param_id_to_buffer,
):
    """Eval model."""
    if args.enable_memory_profiling > 0:
        memory_status_cpu(msg="before train step")

    model.train()
    if args.parallel_proc_data_processing:
        pool = ProcessPoolExecutor(1)

    dp_rank = smp.dp_rank() if not args.prescaled_batch else smp.rdp_rank()
    dp_size = smp.dp_size() if not args.prescaled_batch else smp.rdp_size()
    data_type = "BERT" if args.use_bert_data else "GPT"

    if args.use_bert_data:
        train_paths = sorted(
            [
                os.path.join(args.training_dir, p)
                for p in os.listdir(args.training_dir)
                if os.path.isfile(os.path.join(args.training_dir, p)) and "training" in p
            ]
        )
    else:
        if args.zipped_data > 0:
            file_extension = ".json.gz"
        else:
            file_extension = ".json"
        train_paths = sorted(
            [
                os.path.join(args.training_dir, p)
                for p in os.listdir(args.training_dir)
                if p.endswith(file_extension)
            ]
        )

    train_dataloader = create_pretraining_dataloader(
        [train_paths[start_train_path_index]],
        args.train_batch_size,
        args.max_context_width,
        seed=args.seed,
        dp_rank=dp_rank,
        dp_size=dp_size,
        shuffle=args.same_seed < 1,
        zipped=args.zipped_data > 0,
        use_last_file_only=args.fast_validation > 0,
        data_type=data_type,
    )

    if args.validation_freq is not None:
        # 모든 검증 예제 로드
        if smp.rank() == 0:
            logging.info("Creating val dataloader")
        if args.use_bert_data:
            val_paths = sorted(
                [
                    os.path.join(args.test_dir, p)
                    for p in os.listdir(args.test_dir)
                    if os.path.isfile(os.path.join(args.test_dir, p)) and "testing" in p
                ]
            )

        else:
            if args.zipped_data > 0:
                file_extension = ".json.gz"
            else:
                file_extension = ".json"
            val_paths = sorted(
                [
                    os.path.join(args.test_dir, p)
                    for p in os.listdir(args.test_dir)
                    if p.endswith(file_extension)
                ]
            )
        val_dataloader = create_pretraining_dataloader(
            val_paths,
            args.val_batch_size,
            args.max_context_width,
            seed=args.seed,
            dp_rank=dp_rank,
            dp_size=dp_size,
            shuffle=True,
            zipped=args.zipped_data > 0,
            use_last_file_only=args.fast_validation > 0,
            data_type=data_type,
        )
        if smp.rank() == 0:
            logging.info("Created val dataloader of size %d.", len(val_dataloader))

    start = time.time()
    throughputs = []
    to_save = {"loss": [], "val_loss": []}
    loss_metric = 0

    def grad_accumulation_boundary(batch_idx):
        return batch_idx % args.gradient_accumulation == args.gradient_accumulation - 1

    def should_record():
        # 전역 rank 0을 포함하는 tp 그룹에 있는 rank만 기록합니다.
        if smp.tp_size() > 1:
            tp_group = smp.get_tp_group()
            return 0 in tp_group

        return smp.rank() == 0

    # 계산을 위한 동일한 시드 설정
    set_seed(args.seed)

    for index in range(start_train_path_index, args.epochs * len(train_paths)):
        next_train_path_index = (index + 1) % len(train_paths)
        curr_train_path_index = index % len(train_paths)

        if total_steps >= args.max_steps:
            break

        if args.parallel_proc_data_processing:
            dataset_future = pool.submit(
                create_pretraining_dataloader,
                [train_paths[next_train_path_index]],
                args.train_batch_size,
                args.max_context_width,
                seed=args.seed,
                dp_rank=dp_rank,
                dp_size=dp_size,
                shuffle=args.same_seed < 1,
                zipped=args.zipped_data > 0,
                use_last_file_only=args.fast_validation > 0,
                data_type=data_type,
            )

        if smp.rank() == 0:
            if args.use_bert_data:
                logging.info(
                    "Reading data from training path %s.", train_dataloader.dataset.input_file
                )
            else:
                logging.info(
                    "Reading data from training path %s.", train_dataloader.dataset.input_paths
                )

        for batch_idx, input_data in enumerate(train_dataloader):
            if batch_idx < start_batch_index:
                if smp.rank() == 0:
                    logging.info(
                        "Resuming from saved batch index %d, skipping batch %d ...",
                        start_batch_index,
                        batch_idx,
                    )
                if start_batch_index == len(train_dataloader):
                    # 파일의 마지막 배치에서 저장하는 경우, 다음 파일에서 읽어옵니다.
                    start_batch_index = 0
                    break
                continue

            start_batch_index = 0

            if args.use_bert_data:
                input_ids, _, attention_mask, _, _ = input_data
            else:
                input_ids, attention_mask = input_data

            if total_steps >= args.max_steps:
                break

            torch.cuda.synchronize()
            step_start = time.time()

            if grad_accumulation_boundary(batch_idx - 1):
                optimizer.zero_grad(set_to_none=True)

            if args.logits_output:
                train_output = train_step(model, input_ids, attention_mask, args)
                loss_mb = train_output["loss"]
                logits_mb = train_output["logits"]
                if smp.tp_size() > 1:
                    logits = torch.cat(tuple(logits_mb.outputs), dim=1)  # pylint: disable=no-member
                else:
                    logits = torch.cat(tuple(logits_mb.outputs), dim=0)  # pylint: disable=no-member
            else:
                # 반환 값, loss_mb는 StepOutput 객체입니다.
                loss_mb = train_step(model, input_ids, attention_mask, args)

            # smdistributed: microbatches 간 손실을 평균냅니다.
            loss = loss_mb.reduce_mean()
            if not args.validation_freq:
                loss_metric = loss.item()

            if args.enable_memory_profiling > 0:
                memory_status_cpu("After_train_step_cpu")
                memory_status(msg="After_train_step")

            if args.clean_cache > 0:
                # OOM을 피하기 위해 캐시를 비웁니다.
                torch.cuda.empty_cache()

            if grad_accumulation_boundary(batch_idx):
                if args.sharded_data_parallel_degree < 1:
                    # SDP는 sdp_gradient_clipping 인수를 통해 자체 클리핑을 수행
                    optimizer.clip_master_grads(args.grad_clip)

                optimizer.step()
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()

                if args.enable_memory_profiling > 0:
                    memory_status(msg="After_opt_step")

            torch.cuda.synchronize()
            if args.log_param_norms and args.sharded_data_parallel_degree > 1:
                log_param_norms(model, optimizer, param_id_to_buffer)
            total_steps += 1
            time_elapsed = time.time() - start
            step_time = time.time() - step_start
            sample_processed = input_ids.shape[0] * dp_size
            throughput = sample_processed / step_time
            throughputs.append(throughput)

            # 공식 기반
            # https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/
            tflops_per_gpu = compute_tflops(
                throughput, num_params, smp.size(), input_ids.shape[1], log = (batch_idx == 0)
            )

            if not total_steps % args.logging_freq and args.log_reduced_training_loss > 0:
                loss_detached = loss.detach()
                torch.distributed.all_reduce(loss_detached, group=smp.get_dp_process_group())
                loss_scalar = loss_detached.item() / smp.dp_size()
            else:
                loss_scalar = loss.item()

            if smp.rank() == 0 and not total_steps % args.logging_freq:
                if args.sharded_data_parallel_degree > 1:
                    gradnorm_str = f", Grad norm: {optimizer._global_grad_norm}"
                else:
                    gradnorm_str = ""
                logging.info(
                    "(%ds), Batch %d Loss: %s, Speed: %s samples/sec, TFLOPS/GPU: %s %s",
                    int(time_elapsed),
                    total_steps - 1,
                    loss_scalar,
                    throughput,
                    tflops_per_gpu,
                    gradnorm_str,
                )

                # 초기 단계의 높은 변동성을 제거하기 위해 30단계 후
                # 평균 처리량과 TFLOPs를 계산합니다.
                if len(throughputs) > 30:
                    avg_throughput = np.average(throughputs[30:])
                    avg_tflops = compute_tflops(
                        avg_throughput, num_params, smp.size(), input_ids.shape[1]
                    )
                    logging.info(
                        f"Batch {total_steps - 1},"
                        + f" Running Avg Speed: {avg_throughput} samples/sec,"
                        + f" Running Avg TFLOPS/GPU: {avg_tflops}"
                    )
            # 검증에서 평가합니다.
            if args.validation_freq and not total_steps % args.validation_freq:
                # GPT-NeoX에서 SDPTP를 사용하여 검증을 실행하면 캐시를 정리 해야 합니다.
                torch.cuda.empty_cache()
                cur_state = np.random.get_state()
                model = model.eval()
                val_loss, val_ppl = eval_model(
                    model, val_dataloader, args.validation_batches, args.use_bert_data
                )
                if is_main_process(smp.rank()):
                    logging.info(
                        "(%ds) Batch %d Validation loss: %s",
                        int(time.time() - start),
                        total_steps - 1,
                        val_loss,
                    )
                    logging.info(
                        "(%ds) Batch %d Validation perplexity: %s",
                        int(time.time() - start),
                        total_steps - 1,
                        val_ppl,
                    )
                loss_metric = val_loss
                if args.logits_output:
                    to_save["val_loss"].append(val_loss)
                model = model.train()
                if args.preserve_np_state > 0:
                    np.random.set_state(cur_state)

            # 체크포인트
            if not total_steps % args.checkpoint_freq:
                user_content = {
                    "cli_args": args.__dict__,
                    "num_params": num_params,
                    "total_steps": total_steps,
                    "start_train_path_index": curr_train_path_index,
                    "model_config": model_config,
                    "start_batch_index": batch_idx + 1,
                }

                user_content["lr_scheduler"] = lr_scheduler.state_dict()
                # buffer_names와 param_shapes는 부분 체크포인트를 위해 
                # smp.save_checkpoint()에 의해 자동으로 user_content에 저장됩니다.
                # 이를 통해 전체 모델을 재구성할 수 있습니다.
                smp.save_checkpoint(
                    args.checkpoint_dir,
                    tag=f"total_steps{total_steps}",
                    partial=True,
                    model=model,
                    optimizer=optimizer,
                    user_content=user_content,
                    num_kept_partial_checkpoints=args.num_kept_checkpoints,
                )

            if args.logits_output:
                to_save["loss"].append(loss.item())

        if total_steps >= args.max_steps:
            if should_record() and args.logits_output:
                to_save["logits"] = logits.detach().cpu()
                output_file = f"rank_{smp.rank()}_" + args.logits_output
                torch.save(to_save, os.path.join(args.model_dir, output_file))
                logging.info(
                    "logits and loss saved at %s", os.path.join(args.model_dir, output_file)
                )
            break

        del train_dataloader

        if args.parallel_proc_data_processing:
            s = time.time()  # pylint: disable=invalid-name
            train_dataloader = dataset_future.result(timeout=None)
            wait_time = time.time() - s
            if wait_time > 1:
                # TODO: 이 문제가 발생하면 데이터로더에서 num_workers>1을 시도해야 합니다.  # pylint: disable=fixme
                logging.info(
                    "[%d] Waited %s for data loader to be ready. "
                    "Please check if dataloader performance can be "
                    "improved to avoid these waits.",
                    smp.rank(),
                    wait_time,
                )
        else:
            train_dataloader = create_pretraining_dataloader(
                [train_paths[next_train_path_index]],
                args.train_batch_size,
                args.max_context_width,
                seed=args.seed,
                dp_rank=dp_rank,
                dp_size=dp_size,
                shuffle=args.same_seed < 1,
                zipped=args.zipped_data > 0,
                use_last_file_only=args.fast_validation > 0,
                data_type=data_type,
            )

    # 모든 단계에서 중앙값 처리량을 사용하는 것이 더 견고할 수 있습니다.
    return total_steps, np.median(throughputs) if throughputs else 0, loss_metric


def parse_args():  # pylint: disable=too-many-statements
    """인자 분석"""
    parser = argparse.ArgumentParser()

    # 클라이언트가 보낸 하이퍼파라미터는 스크립트에 명령행 인수로 전달됩니다.

    opt_grp = parser.add_argument_group(
        title="optimization", description="arguments for optimization"
    )
    opt_grp.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="batch size per dp rank, for tensor parallelism degree 8 with pipeline parallel degree 1 this means 8*this batch size per node",  # pylint: disable=line-too-long
    )
    opt_grp.add_argument("--val_batch_size", type=int, default=4)
    opt_grp.add_argument("--max_steps", "--max_training_steps", type=int, default=5000)
    opt_grp.add_argument("--seed", type=int, default=12345)
    opt_grp.add_argument("--same_seed", type=int, default=0)
    opt_grp.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    opt_grp.add_argument("--fp16", default=0, type=int, help="automatic mixed precision training")
    opt_grp.add_argument("--bf16", default=0, type=int, help="automatic mixed precision training")
    opt_grp.add_argument("--sharded_data_parallel_degree", default=1, type=int)
    opt_grp.add_argument("--ddp_dist_backend", type=str, default="auto")
    opt_grp.add_argument("--grad_clip", default=1.0, type=float, help="gradient clipping")
    opt_grp.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
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
    parser.add_argument(
        "--logging_freq", type=int, default=1, help="number of iterations between logging"
    )
    parser.add_argument(
        "--log_param_norms",
        type=int,
        default=0,
        help="to log param norms with logging_freq frequency, currently works only for sharded data parallel jobs",  # pylint: disable=line-too-long
    )
    parser.add_argument(
        "--log_reduced_training_loss",
        type=int,
        default=0,
        help="to log training loss after reducing across all data parallel ranks with logging_freq frequency",  # pylint: disable=line-too-long
    )

    # I/O
    io_grp = parser.add_argument_group(title="io", description="location for input and output")
    io_grp.add_argument("--use_bert_data", type=int, default=0, help="use bert data for training")
    io_grp.add_argument("--zipped_data", type=int, default=1, help="input data is zipped files")
    io_grp.add_argument(
        "--epochs", type=int, default=3, help="times of iterating over the training dataset"
    )
    io_grp.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    io_grp.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/opt/ml/checkpoints",
        help="Saves partial checkpoints (model, optimizer) to this dir, and loads latest checkpoint from this if load_partial is specified.",  # pylint: disable=line-too-long
    )
    io_grp.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="Saves full model for inference to this dir. Also used if load_full is given to load the model. Note the lack of optimizer state here.",  # pylint: disable=line-too-long
    )
    io_grp.add_argument("--training-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    io_grp.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    io_grp.add_argument(
        "--parallel_proc_data_processing",
        type=int,
        default=0,
        help="Load data in parallel with a different process. At any point a process can have two files in memory. With tensor parallelism, each of the 8 processes on an instance will then have 2 files in memory. Depending on file sizes this may or may not be feasible. With pipeline parallelism this was not a problem as only 1 rank on an instance loaded data.",  # pylint: disable=line-too-long
    )
    io_grp.add_argument(
        "--save_final_full_model",
        type=int,
        default=0,
        help="Enabling this will save a combined model only at the end",
    )
    io_grp.add_argument("--load_partial", type=int, default=0, help="Load from partial checkpoints")
    io_grp.add_argument("--load_full", type=int, default=0, help="Load from full checkpoints")
    io_grp.add_argument(
        "--logits_output", type=str, default="", help="Path to save logits and loss"
    )
    io_grp.add_argument("--prescaled_batch", type=int, default=1, help="use prescaled batch")
    # 모델 크기 구성
    model_grp = parser.add_argument_group(
        title="model", description="arguments to describe model configuration"
    )
    model_grp.add_argument(
        "--fine_tune",
        type=int,
        default=0,
        help="Fine-tune model from checkpoint or pretrained model",
    )
    model_grp.add_argument("--model_name", type=str, default="", help="HF model name")
    model_grp.add_argument("--max_context_width", type=int, default=1024)
    model_grp.add_argument("--vocab_size", type=int, default=50264)
    model_grp.add_argument("--hidden_width", type=int, default=768)
    model_grp.add_argument("--intermediate_size", type=int, default=2048)
    model_grp.add_argument("--num_layers", type=int, default=12)
    model_grp.add_argument("--num_heads", type=int, default=12)
    model_grp.add_argument("--num_heads_kv", type=int, default=8)
    model_grp.add_argument("--resid_pdrop", type=float, default=0.1)
    model_grp.add_argument("--embd_pdrop", type=float, default=0.1)
    model_grp.add_argument("--attn_pdrop", type=float, default=0.1)
    model_grp.add_argument("--alibi", type=float, default=0)
    model_grp.add_argument("--summary_first_pdrop", type=float, default=0.1)
    model_grp.add_argument("--use_adamw", type=int, default=0, help="Use adamw optimizer")
    model_grp.add_argument(
        "--use_distributed_transformer", type=int, default=1, help="Use distributed transformer"
    )
    model_grp.add_argument(
        "--checkpoint_sublayers",
        type=int,
        default=0,
        help="Apply activation checkpointing to submodules of each transformer layer",
    )
    model_grp.add_argument("--initializer_range", type=float, default=0.02)

    smp_grp = parser.add_argument_group(title="smp", description="smp")
    smp_grp.add_argument("--tensor_parallel_degree", type=int, default=1)
    smp_grp.add_argument("--pipeline_parallel_degree", type=int, default=1)
    smp_grp.add_argument("--microbatches", type=int, default=1)
    smp_grp.add_argument("--active_microbatches", type=int, default=None)
    smp_grp.add_argument("--optimize", type=str, default="speed")
    smp_grp.add_argument("--activation_strategy", type=str, default="each")
    smp_grp.add_argument("--shard_optimizer_state", type=int, default=0)
    smp_grp.add_argument("--offload_activations", type=int, default=0)
    smp_grp.add_argument("--fast_mode", type=int, default=0)
    smp_grp.add_argument("--static_mode", type=int, default=0)
    smp_grp.add_argument("--delayed_param", type=int, default=0)
    smp_grp.add_argument("--same_partition_load", type=int, default=0)
    smp_grp.add_argument(
        "--attention_in_fp32",
        type=int,
        default=0,
        help="When using FP16 and if the activations overflow, doing the attention computation in fp32 may help. But note that this can substantially increase memory usage and reduce performance. We recommend using bf16 instead which is more numerically stable and would not need this.",  # pylint: disable=line-too-long
    )
    smp_grp.add_argument(
        "--residual_addition_in_fp32",
        type=int,
        default=0,
        help="When using FP16 and if the activations overflow, adding residuals in fp32 may help. But note that this can substantially increase memory usage and reduce performance. We recommend using bf16 instead which is more numerically stable and would not need this.",  # pylint: disable=line-too-long
    )
    smp_grp.add_argument("--placement_strategy", type=str, default="cluster")
    smp_grp.add_argument("--activation_loading_horizon", type=int, default=4)
    smp_grp.add_argument("--skip_tracing", type=int, default=0)
    smp_grp.add_argument("--query_key_layer_scaling", type=int, default=0)
    smp_grp.add_argument("--fused_softmax", type=int, default=1)
    smp_grp.add_argument("--flash_attention", type=int, default=1)
    smp_grp.add_argument("--fused_dropout", type=int, default=0)
    smp_grp.add_argument("--fused_bias_gelu", type=int, default=1)
    smp_grp.add_argument("--gradient_accumulation", type=int, default=1)
    smp_grp.add_argument("--model_type", type=str, default="gpt2")
    smp_grp.add_argument("--rotary_pct", type=float, default=0.25)
    smp_grp.add_argument("--rotary_emb_base", type=int, default=10000)

    parser.add_argument(
        "--num_kept_checkpoints",
        type=int,
        default=5,
        help="how many checkpoints to keep before deleting",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10000,
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
    parser.add_argument(
        "--manual_partition",
        type=int,
        default=0,
        help="evenly distribute layers across the partitions",
    )
    parser.add_argument(
        "--partition_assignment",
        type=str,
        default="",
        help="number of transformer layers assigned to each partition",
    )
    parser.add_argument(
        "--preserve_np_state",
        type=int,
        default=0,
        help="Perserve the numpy random state between validation",
    )
    parser.add_argument(
        "--fast_validation",
        type=int,
        default=1,
        help="Running validation only with the last data file for faster speed",
    )
    parser.add_argument(
        "--gather_if_shard",
        type=int,
        default=1,
        help="When sharding opt states is enabled, gather the opt checkpoint to rdp rank 0 during saving",  # pylint: disable=line-too-long
    )
    parser.add_argument(
        "--clean_cache",
        type=int,
        default=0,
        help="Clean torch reserved memory at he end of every step",
    )
    parser.add_argument("--use_fsx", type=int, default=0, help="Using FSx for checkpointing")
    parser.add_argument(
        "--enable_memory_profiling", type=int, default=0, help="Enable memory profile"
    )

    # 학습률
    lr_grp = parser.add_argument_group(
        title="lr", description="arguments for learning rate schedule"
    )
    lr_grp.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    lr_grp.add_argument(
        "--lr_decay_style",
        type=str,
        default="linear",
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
        default=0.0,
        help="Minumum value for learning rate. The scheduler" "clip values below this threshold.",
    )
    lr_grp.add_argument(
        "--warmup",
        type=float,
        default=0.01,
        help="Percentage of total iterations to warmup on "
        "(.01 = 1 percent of all training iters).",
    )
    lr_grp.add_argument(
        "--plateau",
        type=float,
        default=0.4,
        help="Percentage of total iterations to keep at max if using plateau lr",
    )

    ci_grp = parser.add_argument_group(title="ci", description="ci related settings")
    ci_grp.add_argument("--ci", default=False, action="store_true", help="Whether enable ci")
    ci_grp.add_argument("--time_to_train", type=int, help="time to train threshold")
    ci_grp.add_argument("--throughput", type=float, help="throughput threshold")
    ci_grp.add_argument("--loss", type=float, help="loss threshold")
    args, _ = parser.parse_known_args()
    return args


def compute_num_params(model):
    """매개변수 수 가져오기"""
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


def compute_tflops(throughput, num_params, num_gpus, seq_len, log = False):
    """TFLOPs 계산"""
    tflops = 8 * throughput * num_params / num_gpus * seq_len * 1e-12
    if log and smp.rank() == 0:
        logging.info("Compute tflops: (%s, %s, %s, %s) ==> %s.",
                     throughput, num_params, num_gpus, seq_len, tflops)

    return tflops


def _show_env_vars(rank: Optional[int] = 0):
    env_var = os.environ
    if rank is None or smp.rank() == rank:
        logging.info("Env variables (len = %d):", len(env_var))

        count = 0
        for key, value in sorted(env_var.items()):
            logging.info("  env [%03d/%03d] %-20s: `%s`", count, len(env_var), key, value)
            count += 1


def main():  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """GPT 학습을 위한 주요 함수"""
    args = parse_args()

    if args.partition_assignment != "" and args.manual_partition == 0:
        logging.warning("Partition_assignment is set, enable manual_partition.")
        args.manual_partition = 1

    # any value here is overriden by the config set in notebook when launching the sagemaker job
    smp_config = {
        "ddp": True,
        "tensor_parallel_degree": args.tensor_parallel_degree,
        "pipeline_parallel_degree": args.pipeline_parallel_degree,
        "microbatches": args.microbatches,
        "shard_optimizer_state": args.shard_optimizer_state > 0,
        "prescaled_batch": args.prescaled_batch > 0,
        "fp16": args.fp16 > 0,
        "bf16": args.bf16 > 0,
        "offload_activations": args.offload_activations > 0,
        "delayed_parameter_initialization": args.delayed_param > 0,
        "optimize": args.optimize,
        "placement_strategy": args.placement_strategy,
        "activation_loading_horizon": args.activation_loading_horizon,
        "skip_tracing": args.skip_tracing > 0,
        "auto_partition": not args.manual_partition,
        "default_partition": 0,
        "static_mode": args.static_mode > 0,
        "fast_mode": args.fast_mode > 0,
        "sharded_data_parallel_degree": args.sharded_data_parallel_degree,
        "ddp_dist_backend": args.ddp_dist_backend,
        "sdp_hierarchical_allgather": False,
        "sdp_gradient_clipping": args.grad_clip,
    }
    if args.active_microbatches is not None:
        smp_config["active_microbatches"] = args.active_microbatches
    if args.log_param_norms and args.use_distributed_transformer == 1:
        logging.warning(
            "Script currently doesn't support logging param norms when using distributed transformer, disabling log_param_norms"  # pylint: disable=line-too-long
        )
    smp.init(smp_config)

    _show_env_vars(0)

    if smp.rank() == 0:
        logging.info("Arguments: %s", args.__dict__)
        logging.info("Transformers version: %s", transformers.__version__)
        logging.info(
            "smdistributed.modelparallel version: %s", smdistributed.modelparallel.__version__
        )
        logging.info("smdistributed config: %s", smp_config)

    if args.save_final_full_model and smp.rank() == 0:
        logging.warning(
            "Note that save_final_full_model only saves the final model at the end "
            "of all steps. It does not save optimizer state. Optimizer state is only "
            "saved with partial models which are saved at checkpointing_freq during "
            "training. If you want to restart training you need partial checkpoints."
        )

    if args.partition_assignment != "":
        partition_assignment = args.partition_assignment.split(",")
        msg = (
            f"partition_assignment must have the same size as pipeline parallel degree, "
            f"but getting {len(partition_assignment)} vs {smp.pp_size()}"
        )
        logging.fatal("Will fail with: %s.", msg)
        raise AssertionError(msg)

    model_config = AutoConfig.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
    model_config.hidden_size = args.hidden_width
    model_config.n_layer = args.num_layers
    model_config.n_head = args.num_heads
    model_config.n_head_kv = args.num_heads_kv
    model_config.use_cache = False

    # 다음 방법은 원래 모델에서 가중치를 올바르게 초기화하는 것을 건너뛰어 시작 시간을 단축합니다.
    # 이는 'DistributedModel'이 분산 트랜스포머를 사용할 때 가중치를 덮어쓸 것이기 때문에 문제가 되지 않습니다.    
    if args.use_distributed_transformer > 0:
        from transformers.modeling_utils import (  # pylint: disable=import-error,import-outside-toplevel
            PreTrainedModel,
        )

        PreTrainedModel.init_weights = lambda x: None

    set_seed(args.seed)

    if args.enable_memory_profiling > 0:
        memory_status_cpu(msg="before model creation")

    if args.fp16 and args.bf16:
        raise ValueError("FP16 and BF16 cannot be simultaneously enabled.")

    if args.fp16:
        dtype = torch.float16  # pylint: disable=no-member
    elif args.bf16:
        dtype = torch.bfloat16  # pylint: disable=no-member
    else:
        dtype = torch.get_default_dtype()  # pylint: disable=no-member

    if args.fine_tune > 0 and smp.rank() == 0:
        if args.model_type == "flan_t5":
            pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name or args.model_dir
            )
        else:
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                args.model_name or args.model_dir
            )
        model_state_dict = pretrained_model.state_dict()
        path = os.path.join(args.model_dir, "fullmodel.pt")
        torch.save(model_state_dict, path)
    smp.barrier()

    # zero_init에 대한 설명:
    # 실제 모델의 학습을 위해 0으로 초기화하고 싶습니다.
    # disttf 경우 DistModel 래퍼에서 사용됩니다. 다른 경우에는 0 초기화를 설정할 필요가 없습니다.
    # 이것은 오직 param_id_to_offset에 필요합니다.    
    with smp.model_creation(
        tensor_parallelism=smp.tp_size() > 1 or args.use_distributed_transformer > 0,
        zero_init=args.use_distributed_transformer == 0,
        dtype=dtype,
        distribute_embedding=args.sharded_data_parallel_degree > 1 and smp.tp_size() > 1,
        use_alibi=args.alibi > 0,
        attention_in_fp32=args.attention_in_fp32 > 0,
        fp32_residual_addition=args.residual_addition_in_fp32 > 0,
        query_key_layer_scaling=args.query_key_layer_scaling > 0 and args.bf16 < 1,
        fused_softmax=args.fused_softmax > 0,
        fused_dropout=args.fused_dropout > 0,
        fused_bias_gelu=args.fused_bias_gelu > 0,
        flash_attention=args.flash_attention > 0,
    ):
        if args.model_type == "flan_t5":
            model = AutoModelForSeq2SeqLM.from_config(model_config)
        else:
            model = RWForCausalLM(model_config)

    if args.enable_memory_profiling > 0:
        memory_status_cpu(msg="after model creation")

    # smdistributed: 현재 프로세스에서 사용 중인 GPU ID로 장치를 설정합니다.
    # 입력 텐서는 이 장치로 전송되어야 합니다.
    torch.cuda.set_device(smp.local_rank())

    if not args.same_seed:
        # tp_rank에 따라 시드를 설정하여 서로 다른 tp_rank에서 가중치가 동일해지는 것을 방지합니다.
        set_seed(args.seed + smp.tp_rank())

    # smdistributed: DistributedModel 컨테이너를 사용하여 모델을 다양한 rank에 걸쳐 분할합니다. 
    # 스크립트의 나머지 부분에서는 DistributedModel 클래스 인스턴스화에 제공된 모델 대신
    # 반환된 DistributedModel 객체를 사용해야 합니다.    
    if args.enable_memory_profiling > 0:
        memory_status_cpu(msg="before dist model creation")

    model = smp.DistributedModel(
        model, trace_device="gpu", backward_passes_per_step=args.gradient_accumulation
    )

    if args.enable_memory_profiling > 0:
        memory_status_cpu(msg="after dist model creation")
    m = model.get_module()  # pylint: disable=invalid-name

    num_params = compute_num_params(m)
    if smp.rank() == 0:
        logging.info("# total parameters: %s", num_params)

    if args.use_distributed_transformer > 0:
        transformer_layers = m.transformer.seq_layers
    else:
        if args.model_type in ["gpt2", "bloom", "falcon"]:
            transformer_layers = m.transformer.h
        elif args.model_type == "gpt_neox":
            transformer_layers = m.gpt_neox.layers
        elif args.model_type == "flan_t5":
            transformer_layers = m.encoder.block

    if args.manual_partition:
        logging.debug("Manual partition enabled")
        if args.partition_assignment != "":
            get_num_layers = lambda x: int(  # pylint: disable=unnecessary-lambda-assignment
                partition_assignment[x]
            )
            total_layers = sum(get_num_layers(pp_rank) for pp_rank in range(smp.pp_size()))

            msg = (
                f"partition_assignment must have the same total transformer layers as model, "
                f"but getting {total_layers} vs {args.num_layers}"
            )
            logging.fatal("Will fail with: %s.", msg)
            raise AssertionError(msg)

        # 모든 파티션에 레이어를 균등하게 분산합니다.
        div, rem = divmod(args.num_layers, smp.pp_size())
        get_num_layers = lambda x: (  # pylint: disable=unnecessary-lambda-assignment
            div + 1 if x >= smp.pp_size() - rem else div
        )

        assignments = []
        # (TODO) 175B에 필요하며 그렇지 않으면 파티션 "8,17,17,18,18,18"에서 멈추는 문제가 발생합니다.
        # 추가 조사 필요
        # for pp_rank in reversed(range(smp.pp_size())):
        for pp_rank in range(smp.pp_size()):
            nl = get_num_layers(pp_rank)  # pylint: disable=invalid-name
            logging.debug("%s layers assigned to partition %d", nl, pp_rank)
            assignments += [pp_rank for _ in range(nl)]

        for i, c in enumerate(transformer_layers.children()):  # pylint: disable=invalid-name
            smp.set_partition(c, assignments[i])

    param_groups = get_param_groups_by_weight_decay(m)

    if args.use_adamw > 0:
        optimizer = optim.AdamW(
            param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.Adam(
            param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay
        )

    if args.activation_checkpointing:  # pylint: disable=too-many-nested-blocks
        if args.use_distributed_transformer or smp.tp_size() > 1:
            if args.checkpoint_sublayers:
                for c in transformer_layers.children():  # pylint: disable=invalid-name
                    smp.set_activation_checkpointing(c.attention)
                    smp.set_activation_checkpointing(c.output)
            else:
                smp.set_activation_checkpointing(
                    transformer_layers, strategy=args.activation_strategy
                )
        else:
            for c in transformer_layers.children():  # pylint: disable=invalid-name
                if args.checkpoint_sublayers:
                    if args.model_type == "gpt2":
                        smp.set_activation_checkpointing(c.attn)
                        smp.set_activation_checkpointing(c.mlp)
                    elif args.model_type in ["gpt_neox", "bloom"]:
                        if args.model_type == "gpt_neox":
                            smp.set_activation_checkpointing(c.attention)
                        elif args.model_type == "bloom":
                            smp.set_activation_checkpointing(c.self_attention)
                        smp.set_activation_checkpointing(c.input_layernorm)
                        smp.set_activation_checkpointing(c.post_attention_layernorm)
                        smp.set_activation_checkpointing(c.mlp)
                    elif args.model_type == "falcon":
                        smp.set_activation_checkpointing(c.self_attention)
                else:
                    smp.set_activation_checkpointing(c)
            # T5에서 디코더 레이어 체크포인트
            if args.model_type == "flan_t5":
                for c in m.decoder.block.children():  # pylint: disable=invalid-name
                    smp.set_activation_checkpointing(c)

    if args.log_param_norms > 0 and args.sharded_data_parallel_degree > 1 and args.use_distributed_transformer == 0:
        param_id_to_offset = build_param_id_to_offset(param_groups)

    optimizer = smp.DistributedOptimizer(
        optimizer,
        static_loss_scale=None,
        dynamic_loss_scale=True,
        dynamic_loss_args={"scale_window": 1000, "min_scale": 1, "delayed_shift": 2},
    )

    if args.fine_tune > 0:
        smp.resume_from_checkpoint(args.model_dir, tag="fullmodel.pt", partial=False)

    if args.log_param_norms > 0 and args.sharded_data_parallel_degree > 1 and args.use_distributed_transformer == 0:
        param_id_to_buffer = build_param_id_to_buffer(optimizer, param_id_to_offset)
    else:
        param_id_to_buffer = None

    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.enable_memory_profiling > 0:
        model.register_post_partition_hook(
            lambda model, optimizer: memory_status(msg="After partition")
        )

    # 모델과 옵티마이저를 분산된 smp로 래핑한 후 로드합니다...
    if args.load_full or args.load_partial:
        if args.load_partial and args.load_full:
            logging.info(
                "Since both --load_partial and --load_full set, will try to load from full "
                "checkpoint. If the intention is to load from partial checkpoint, please don't set "
                "--load_full"
            )
        partial = not args.load_full
        path = args.checkpoint_dir if partial else args.model_dir
        tag = None if partial else "fullmodel.pt"
        user_content = smp.resume_from_checkpoint(path, tag=tag, partial=partial)
        total_steps = user_content["total_steps"] if partial else 0
        start_train_path_index = user_content.get("start_train_path_index", 0)
        start_batch_index = user_content.get("start_batch_index", 0)
        if "lr_scheduler" in user_content:
            lr_scheduler.load_state_dict(user_content["lr_scheduler"])
    else:
        total_steps = 0
        start_train_path_index = 0
        start_batch_index = 0

    # 부분 체크포인팅과 SDPTP 및 GPT NeoX의 경우 메모리를 지우기 위해 빈 캐시를 추가합니다.
    torch.cuda.empty_cache()

    start = time.time()
    total_steps, throughput, loss = train(
        model,
        optimizer,
        lr_scheduler,
        model_config,
        start_train_path_index,
        start_batch_index,
        num_params,
        total_steps,
        args,
        param_id_to_buffer,
    )
    time_to_train = time.time() - start
    if args.ci:
        logging.info("[SMP_METRIC]__T5__Time_to_train__%s", time_to_train)
        logging.info("[SMP_METRIC]__T5__samples/second__%s", throughput)
        logging.info("[SMP_METRIC]__T5__Loss__%s", loss)
        if not args.load_partial and not args.load_full:
            if time_to_train >= args.time_to_train:
                msg = f"Time to train ({time_to_train}) >= threshold ({args.time_to_train})"
                logging.fatal("Will fail with: %s.", msg)
                raise AssertionError(msg)

            if throughput <= args.throughput:
                msg = f"Throughput ({throughput}) >= threshold ({args.throughput})"
                logging.fatal("Will fail with: %s.", msg)
                raise AssertionError(msg)

            if args.loss and loss >= args.loss:
                msg = f"Loss ({loss}) >= threshold ({args.loss})"
                logging.fatal("Will fail with: %s.", msg)
                raise AssertionError(msg)

    if args.save_final_full_model:
        # 끝에서 전체 모델을 저장합니다.
        user_content = {
            "cli_args": args.__dict__,
            "num_params": num_params,
            "total_steps": total_steps,
            "model_config": model_config,
        }
        # pylint: disable=line-too-long
        # 다음 API를 사용하여 SDP 체크포인트에서 전체 모델을 얻을 수도 있습니다.
        # > from smp.sharded_data_parallel_checkpoint import get_full_state_dict_from_sharded_data_parallel_checkpoint
        # > full_model = get_full_state_dict_from_sharded_data_parallel_checkpoint(args.model_dir, tag=f"sharded_data_parallel_final_full_{num_params}", dtype=torch.float32)
        # > if args.use_distributed_transformer > 0: # translate the state_dict to hf format if distributed transformer is used
        # >     full_model = smp.nn.huggingface.gpt2.translate_state_dict_to_hf_gpt2(full_model, max_seq_len=args.max_context_width)
        # 참고: 공유된 매개변수는 반영되지 않으므로 로드 중에는 `strict=False`로 로드해야 할 수 있습니다.
        # pylint: enable=line-too-long
        smp.save_checkpoint(
            args.model_dir,
            tag="fullmodel.pt",
            partial=False,
            model=model,
            user_content=user_content,
        )

    smp.barrier()
    if smp.rank() == 0:
        logging.info("SMP training finished successfully")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s " "[%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    main()
