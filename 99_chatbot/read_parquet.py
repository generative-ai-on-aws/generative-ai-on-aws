from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from datasets import load_dataset

import torch

import numpy as np
import pandas as pd

conversation_dataset_original = load_dataset("parquet", data_files={"train": "/tmp/conversations-2023-05-28-1685316076662.parquet"})
print(conversation_dataset_original)

print(conversation_dataset_original["train"][0])

