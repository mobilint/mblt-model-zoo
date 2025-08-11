Pretrained Models with Huggingface's Transformers
========================

**mblt-model-zoo** also provides generative AI models from Huggingface's [Transformers](https://github.com/huggingface/transformers).
Currently, these models are only available on Mobilint's [Aries](https://www.mobilint.com/aries).
Support for [Regulus](https://www.mobilint.com/regulus) is planned and currently under development

Mobilint's Model Zoo provides a seamless experience for using Transformers' models with the same class/function interfaces. All of the necessary auto classes in Transformers are overridden with our own, which automatically detect our models' identifiers (e.g., `mobilint/Llama-3.2-3B-Instruct`) and download the required files from our model server. It also supports a locally downloaded model directory, just like the original Transformers.

## Installation
- Install Mobilint ACCELerator(MACCEL) on your environment. In case you are not Mobilint customer, please contact [us](mailto:tech-support@mobilint.com).
- Install **mblt-model-zoo** with extra dependency using pip:
```bash
pip install mblt-model-zoo[transformers]
```
- If you want to install the latest version from source, clone the repository and install it:
```bash
git clone https://github.com/mobilint/mblt-model-zoo.git
cd mblt-model-zoo
pip install -e .[transformers]
```

## Quick Start Guide

### Working with Quantized Model

**mblt-model-zoo** provides quantized models based on Transformers with the same interfaces. You can use our overrided auto classes such as `pipeline`, `AutoModel`, and `AutoTokenizer` with our models' ids. The following code snippet shows how to use the pre-trained model for inference with `pipeline`.

```python
from mblt_model_zoo.transformers import pipeline, AutoTokenizer
from transformers import TextStreamer

model_path = "mobilint/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = pipeline(
    "text-generation",
    model=model_path,
    streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
)

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

outputs = pipe(
    messages,
    max_new_tokens=2048,
)
```

You can also use `AutoModel` or `AutoModelForCausalLM` for initializing models.

```python
from mblt_model_zoo.transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer

model_path = "mobilint/EXAONE-3.5-2.4B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)

outputs = model.generate(
    inputs,
    max_new_tokens=2048,
    do_sample=True,
    top_k=50,
    top_p=0.95
)
```

Other usage examples can be found in the [tutorial](../../tests/tutorial/) directory.

### Listing Available Models

**mblt-model-zoo** offers a function to list all available models. You can use the following code snippet to list the models for a specific task (e.g., `large_language_model`, `speech_to_text`, etc.):

```python
from mblt_model_zoo.transformers import list_models
from pprint import pprint

available_models = list_models()
pprint(available_models)
```

## Model List
The following tables summarize Transformers' models available in **mblt-model-zoo**. We provide the models that are quantized with our advanced quantization techniques. Performance metrics will be provided in the future.

### Large Language Models

| Model | Model ID | Source | Note |
| ----- | -------- | ------ | ---- |
| EXAONE-3.5-2.4B-Instruct | `mobilint/EXAONE-3.5-2.4B-Instruct` | [Link](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct) | |
| EXAONE-4.0-1.2B | `mobilint/EXAONE-4.0-1.2B` | [Link](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B) | |
| EXAONE-Deep-2.4B | `mobilint/EXAONE-Deep-2.4B` | [Link](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-2.4B) | |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | `mobilint/HyperCLOVAX-SEED-Text-Instruct-1.5B` | [Link](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B) | |
| Llama-3.1-8B-Instruct | `mobilint/Llama-3.1-8B-Instruct` | [Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | |
| Llama-3.2-1B-Instruct | `mobilint/Llama-3.2-1B-Instruct` | [Link](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | |
| Llama-3.2-3B-Instruct | `mobilint/Llama-3.2-3B-Instruct` | [Link](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | |
| c4ai-command-r7b-12-2024 | `mobilint/c4ai-command-r7b-12-2024` | [Link](https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024) | |

### Speech-To-Text Models

| Model | Model ID | Source | Note |
| ----- | -------- | ------ | ---- |
| whisper-small | `mobilint/whisper-small` | [Link](https://huggingface.co/openai/whisper-small) | |

### Vision Language Models

| Model | Model ID | Source | Note |
| ----- | -------- | ------ | ---- |
| aya-vision-8b | `mobilint/aya-vision-8b` | [Link](https://huggingface.co/CohereLabs/aya-vision-8b) | |
| blip-image-captioning-large | `mobilint/blip-image-captioning-large` | [Link](https://huggingface.co/Salesforce/blip-image-captioning-large) | |

## License
The Mobilint Model Zoo is released under BSD 3-Clause License. Please see the [LICENSE](https://github.com/mobilint/mblt-model-zoo/blob/master/LICENSE) file for more details.

Additionally, the license for each model provided in this package follows the terms specified in the source link provided with it.

## Support & Issues
If you encounter any problems with this package, please feel free to contact [us](mailto:tech-support@mobilint.com).