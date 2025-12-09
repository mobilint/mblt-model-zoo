import maccel
import torch
from transformers import AutoTokenizer

from mblt_model_zoo.transformers.large_language_model.llama_batch import (
    MobilintLlamaBatchForCausalLM,
)

model_path = "/home/mobilint/.mblt_model_zoo/transformers/Llama-3.1-8B-Instruct-Batch16"

model = MobilintLlamaBatchForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

texts = [
    "This is sample text.",
    "Hello! My name is ",
    "LLM is",
]

inputs = tokenizer(texts)

input_ids = inputs["input_ids"]

embeddings = model.get_input_embeddings()

input_embeds = [embeddings(torch.tensor(input_id)) for input_id in input_ids]

sequence_lengths = [embed.shape for embed in input_embeds]

batch_param = maccel.BatchParam(
    sequence_lengths=sequence_lengths,
    cache_sizes=[0, 0, 0],
    cache_ids=[0, 1, 2],
    prefill_masks=[False, False, False],  # not implemented in C++ yet.
)

output = model.get_cache_mxq_model().infer(
    [torch.concat(input_embeds, dim=0).unsqueeze(0)], None, 0, batch_param
)
output1 = output[0]

print(sequence_lengths)

input_embeds.reverse()

sequence_lengths = [embed.shape for embed in input_embeds]

batch_param = maccel.BatchParam(
    sequence_lengths=sequence_lengths,
    cache_sizes=[0, 0, 0],
    cache_ids=[0, 1, 2],
    prefill_masks=[False, False, False],  # not implemented in C++ yet.
)

output = model.get_cache_mxq_model().infer(
    [torch.concat(input_embeds, dim=0).unsqueeze(0)], None, 0, batch_param
)
output2 = output[0]

print(sequence_lengths)

print(output1)
print(output2)
