# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import numpy as np
import datasets as hf_datasets
from torch.utils.data import IterableDataset
from typing import Dict, Iterable, Union
from transformers import PreTrainedTokenizerBase

class ConcatTokensDataset(IterableDataset):
    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        wrap: bool,
    ):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.should_wrap = wrap

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        buffer = []
        mask_buffer = []
        for sample in self.hf_dataset:
            encoded = self.tokenizer(sample['text'],
                                     truncation=False,
                                     padding=False)
            iids = encoded['input_ids']
            mask = encoded['attention_mask']
            buffer = buffer + iids + [self.tokenizer.eos_token_id]
            mask_buffer = mask_buffer + mask + [1]
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:] if self.should_wrap else []
                concat_sample_mask = mask_buffer[:self.max_length]
                mask_buffer = mask_buffer[self.max_length:] if self.should_wrap else []
                yield np.array(concat_sample)
