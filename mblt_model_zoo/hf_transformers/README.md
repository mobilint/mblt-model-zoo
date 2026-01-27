# Pretrained Models with Huggingface's Transformers

**mblt-model-zoo** also provides generative AI models from Huggingface's [Transformers](https://github.com/huggingface/transformers).
Currently, these models are only available on Mobilint's [ARIES](https://www.mobilint.com/aries).
Support for [REGULUS](https://www.mobilint.com/regulus) is planned and currently under development

Mobilint's Model Zoo provides a seamless experience for using `transformers` models with the same class/function interfaces. All of the auto classes in `transformers` can import our pre-quantized models (e.g., `mobilint/Llama-3.2-3B-Instruct`) and download the required files from Huggingface hub. It also supports a locally downloaded model directory, just like the original `transformers`.

## Installation

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

**mblt-model-zoo** provides quantized models based on Transformers with the same interfaces. If `mblt-model-zoo` package is installed, you can use auto classes from `transformers` such as `pipeline`, `AutoModel`, and `AutoTokenizer` with our models' ids. The following code snippet shows how to use the pre-trained model for inference with `pipeline`. Our models include proxy python codes to import needed config and model classes. So, `trust_remote_code=True` option is needed.

Some models require a specific model revision (e.g., `W8`). Pass `revision="W8"` to `from_pretrained`/`pipeline` to select the desired quantized variant.

### Quantization Revision Terms

We use the following revision labels for quantized variants:
- `W8`: all weights are quantized to INT8.
- `W4`: all weights are quantized to INT4.
- `W4V8`: in the attention QKV matrices, the Value (V) matrix is INT8, and the rest are INT4.

```python
from transformers import TextStreamer, pipeline, AutoTokenizer

model_path = "mobilint/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, revision="W8")

pipe = pipeline(
    "text-generation",
    model=model_path,
    streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
    trust_remote_code=True,
    revision="W8",
    device="cpu",
)

pipe.generation_config.max_new_tokens = None

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

outputs = pipe(
    messages,
    max_length=4096,
)
```

You can also use `AutoModel` or `AutoModelForCausalLM` for initializing models.

```python
from transformers import TextStreamer, AutoModelForCausalLM, AutoTokenizer

model_path = "mobilint/EXAONE-3.5-2.4B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, revision="W8")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    revision="W8",
).to("cpu")

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

streamer = TextStreamer(tokenizer)
outputs = model.generate(
    input_ids.to(model.device),
    max_new_tokens=2048,
    do_sample=True,
    streamer=streamer,
)
```

We also support the vision-language models associated with `AutoProcessor` and image format inputs.

```python
from transformers import TextStreamer, pipeline, AutoProcessor

model_name = "mobilint/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    model_name,
    use_fast=True,
    trust_remote_code=True,
    revision="W8",
)
pipe = pipeline(
    "image-text-to-text",
    model=model_name,
    processor=processor,
    revision="W8",
    device="cpu",
)
pipe.generation_config.max_new_tokens = None

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

pipe(
    text=messages,
    generate_kwargs={
        "max_length": 4096,
        "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
    },
)
```

Further usage examples can be found in the [tests](../../tests/transformers) directory.

## TPS Benchmark CLI (Sweep)

If you installed the optional extra (`pip install mblt-model-zoo[transformers]`), you can run a simple TPS sweep from the command line:

```bash
mblt-model-zoo tps sweep --model mobilint/Llama-3.2-3B-Instruct --device cpu --revision W8 --json tps.json --plot tps.png
```

Use `--revision` to select a specific quantized revision (e.g., `W8`).

### Listing Available Models

**mblt-model-zoo** offers a function to list all available models. You can use the following code snippet to list the models for a specific task (e.g., `text-generation`, `automatic-speech-recognition`, etc.):

```python
from pprint import pprint
from mblt_model_zoo.hf_transformers.utils import list_models

available_models = list_models()
pprint(available_models)
```

It will search online to look up available models. When offline, it will list cached models in the current environment.

## Model List

The following tables summarize Transformers' models available in **mblt-model-zoo**. We provide the models that are quantized with our advanced quantization techniques. Performance metrics will be provided in the future.

### Large Language Models

| Model | Model ID | Source | Note |
| ----- | -------- | ------ | ---- |
| EXAONE-3.5-2.4B-Instruct | `mobilint/EXAONE-3.5-2.4B-Instruct` | [Link](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct) | |
| EXAONE-3.5-7.8B-Instruct | `mobilint/EXAONE-3.5-7.8B-Instruct` | [Link](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct) | |
| EXAONE-4.0-1.2B | `mobilint/EXAONE-4.0-1.2B` | [Link](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B) | |
| EXAONE-Deep-2.4B | `mobilint/EXAONE-Deep-2.4B` | [Link](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-2.4B) | |
| HyperCLOVAX-SEED-Text-Instruct-0.5B | `mobilint/HyperCLOVAX-SEED-Text-Instruct-0.5B` | [Link](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B) | |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | `mobilint/HyperCLOVAX-SEED-Text-Instruct-1.5B` | [Link](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B) | |
| Llama-3.1-8B-Instruct | `mobilint/Llama-3.1-8B-Instruct` | [Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | |
| Llama-3.2-1B-Instruct | `mobilint/Llama-3.2-1B-Instruct` | [Link](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | |
| Llama-3.2-3B-Instruct | `mobilint/Llama-3.2-3B-Instruct` | [Link](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | |
| Qwen2.5-0.5B-Instruct | `mobilint/Qwen2.5-0.5B-Instruct` | [Link](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | |
| Qwen2.5-1.5B-Instruct | `mobilint/Qwen2.5-1.5B-Instruct` | [Link](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | |
| Qwen2.5-3B-Instruct | `mobilint/Qwen2.5-3B-Instruct` | [Link](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | |
| Qwen2.5-7B-Instruct | `mobilint/Qwen2.5-7B-Instruct` | [Link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | |
| bert-base-uncased | `mobilint/bert-base-uncased` | [Link](https://huggingface.co/google-bert/bert-base-uncased) | |
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
| Qwen2-VL-2B-Instruct | `mobilint/Qwen2-VL-2B-Instruct` | [Link](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Only supports 1 image input with (224, 224) size. Image input will be resized automatically by our overrided preprocessor. |

## License

The Mobilint Model Zoo is released under BSD 3-Clause License. Please see the [LICENSE](https://github.com/mobilint/mblt-model-zoo/blob/master/LICENSE) file for more details.

Additionally, the license for each model provided on the Huggingface hub follows the terms specified in each repository.

## Support & Issues

If you encounter any problems with this package, please feel free to contact [us](mailto:tech-support@mobilint.com).
