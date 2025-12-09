import maccel
import numpy as np
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
    "Harry Potter, the famous ",
]

inputs = tokenizer(texts)

input_ids = inputs["input_ids"]

embeddings = model.get_input_embeddings()

input_embeds = [embeddings(torch.tensor(input_id)).detach() for input_id in input_ids]

sequence_lengths = [int(embed.shape[0]) for embed in input_embeds]

batch_param = maccel.BatchParam(
    sequence_lengths=sequence_lengths,
    cache_sizes=[0 for _ in sequence_lengths],
    cache_ids=[i for i in range(len(sequence_lengths))],
    prefill_masks=[False for _ in sequence_lengths],  # not implemented in C++ yet.
)

output = model.get_cache_mxq_model().infer(
    [torch.concat(input_embeds, dim=0).unsqueeze(0).cpu().numpy()], None, 0, batch_param
)
output1 = output[0][0, 0, :, :]

input_embeds.reverse()

sequence_lengths = [int(embed.shape[0]) for embed in input_embeds]

batch_param = maccel.BatchParam(
    sequence_lengths=sequence_lengths,
    cache_sizes=[0 for _ in sequence_lengths],
    cache_ids=[i for i in range(len(sequence_lengths))],
    prefill_masks=[False for _ in sequence_lengths],  # not implemented in C++ yet.
)

output = model.get_cache_mxq_model().infer(
    [torch.concat(input_embeds, dim=0).unsqueeze(0).cpu().numpy()], None, 0, batch_param
)
output2 = output[0][0, 0, :, :]

print(output1[0, :], output2[2, :], output1.shape, output2.shape)

assert np.all(output1[0, :] == output2[2, :])
assert np.all(output1[1, :] == output2[1, :])
assert np.all(output1[2, :] == output2[0, :])
