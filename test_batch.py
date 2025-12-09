from transformers import AutoTokenizer

from mblt_model_zoo.transformers.large_language_model.llama_batch import (
    MobilintLlamaBatchForCausalLM,
)

model_path = "~/.mblt_model_zoo/transformers/Llama-3.1-8B-Instruct-Batch16"

model = MobilintLlamaBatchForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

texts = [
    "This is sample text.",
    "Hello! My name is ",
    "LLM is",
]

inputs = tokenizer(texts)

print(inputs)
