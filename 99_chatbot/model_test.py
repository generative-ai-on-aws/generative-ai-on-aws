from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from datasets import load_dataset

from transformers.models.llama import LlamaModel
from transformers.models.llama import LlamaTokenizerFast

#dataset_name="nyanko7/LLaMA-65B"
model_name="llama-65b"

tokenizer = LlamaTokenizerFast.from_pretrained(model_name) #, device_map="auto")
model = LlamaModel.from_pretrained(model_name) #, device_map="auto")

# load dataset
#dataset = load_dataset(dataset_name, split="train")


