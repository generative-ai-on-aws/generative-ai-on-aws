from functools import partial
from itertools import chain


remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}


# empty list to save remainder from batches to use in next batch
def pack_dataset(dataset, chunk_length=2048):
    print(f"Chunking dataset into chunks of {chunk_length} tokens.")

    def chunk(sample, chunk_length=chunk_length):
        # define global remainder variable to save remainder from batches to use in next batch
        global remainder
        # Concatenate all texts and add remainder from previous batch
        concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
        concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
        # get total number of tokens for batch
        batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

        # get max number of chunks for batch
        if batch_total_length >= chunk_length:
            batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
            for k, t in concatenated_examples.items()
        }
        # add remainder to global variable for next batch
        remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
        # prepare labels
        result["labels"] = result["input_ids"].copy()
        return result

    # tokenize and chunk dataset
    lm_dataset = dataset.map(
        partial(chunk, chunk_length=chunk_length),
        batched=True,
    )
    print(f"Total number of samples: {len(lm_dataset)}")
    return lm_dataset