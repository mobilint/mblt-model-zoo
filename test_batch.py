import torch
from transformers import AutoTokenizer

from mblt_model_zoo.transformers.large_language_model.llama_batch import (
    MobilintLlamaBatchForCausalLM,
)

model_path = "/home/mobilint/.mblt_model_zoo/transformers/Llama-3.1-8B-Instruct-Batch16"

model = MobilintLlamaBatchForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

texts = [
    "This is sample text.",
    "Hello! My name is ",
    "LLM is",
]

inputs = tokenizer(texts)

input_ids = inputs["input_ids"]

embeddings = model.get_input_embeddings()

input_embeds = [embeddings(torch.tensor(input_id)) for input_id in input_ids]

print(input_embeds)
