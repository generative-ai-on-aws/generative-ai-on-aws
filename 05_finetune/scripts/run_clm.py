from dataclasses import dataclass, field

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)

from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronTrainer as Trainer
from optimum.neuron import NeuronTrainingArguments as TrainingArguments
from optimum.neuron.distributed import lazy_load_for_parallelism


def training_function(script_args, training_args):
    # load dataset
    dataset = load_from_disk(script_args.dataset_path)

    # load model from the hub with a bnb config
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    with lazy_load_for_parallelism(tensor_parallel_size=training_args.tensor_parallel_size):
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            use_cache=False if training_args.gradient_checkpointing else True,
        )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,  # no special collator needed since we stacked the dataset
    )

    # Start training
    trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload

    # Consolidate sharded checkpoint files to single file when TP degree > 1
    # perrysc@amazon.com
    # if (int(os.environ.get("RANK", -1)) == 0) and int(training_args.tensor_parallel_size) > 1:
    #     print("Converting sharded checkpoint to consolidated format")
    #     from optimum.neuron.distributed.checkpointing import (
    #         consolidate_tensor_parallel_checkpoints_to_unified_checkpoint,
    #     )
    #     from shutil import rmtree

    #     consolidate_tensor_parallel_checkpoints_to_unified_checkpoint(
    #         training_args.output_dir, training_args.output_dir, "pytorch"
    #     )
    #     rmtree(os.path.join(training_args.output_dir, "tensor_parallel_shards"))  # remove sharded checkpoint files


@dataclass
class ScriptArguments:
    model_id: str = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_path: str = field(
        metadata={"help": "Path to the preprocessed and tokenized dataset."},
        default=None,
    )


def main():
    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()

    # set seed
    set_seed(training_args.seed)

    # run training function
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()