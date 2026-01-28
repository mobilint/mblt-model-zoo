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
    model_kwargs={"embedding_weight": "/path/to/embedding.pt"},
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
    embedding_weight="/path/to/embedding.pt",
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
    model_kwargs={"embedding_weight": "/path/to/embedding.pt"},
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


### Keyword Parameters

When loading models with `from_pretrained`, you can pass extra keyword parameters to customize runtime behavior.
For `pipeline(...)`, pass them via `model_kwargs={...}`.

#### NPU settings

These are custom keyword parameters for Mobilint NPU execution (the compiled model is stored in an `*.mxq` file).

- `mxq_path` (`str`)
    Overrides which `*.mxq` file to load.
    You can pass either a local path or a path within the Hugging Face repository.
    Resolution order is:
    1) If `mxq_path` exists on disk (relative or absolute), it is used as-is.
    2) If `name_or_path` is a local directory, `os.path.join(name_or_path, mxq_path)` is tried.
    3) Otherwise, the loader tries to download `mxq_path` from the Hugging Face Hub (preferring the current `revision`), with fallbacks:
       - retry without `revision`
       - try to reuse a cached `*.mxq` from the local HF cache
       - finally, pick a best-effort `*.mxq` candidate from the repo if the exact path is not found

- `core_mode` (`str`)
    Selects how the NPU runtime schedules work across cores/clusters.
    Supported values:
    - `single`: run on specific cores (use `target_cores`)
    - `multi`: run on one or more clusters (use `target_clusters`)
    - `global4`: global scheduling across 4 cores (use `target_clusters`)
    - `global8`: global scheduling across all cores (requires all clusters)

    Note: the effective/valid core mode depends on how the `*.mxq` was compiled. If you are not sure, keep the default stored in the model config.

- `target_cores` (`list[str]`)
    Used only when `core_mode="single"`. Each entry must be in the form `"cluster:core"`.
    - `cluster`: `0` or `1`
    - `core`: `0`, `1`, `2`, or `3`

    Example: `target_cores=["0:0", "0:1"]`

- `target_clusters` (`list[int]`)
    Used when `core_mode` is `multi`, `global4`, or `global8`. Each entry is a cluster index (`0` or `1`).
    - For `global8`, all clusters must be included (e.g. `target_clusters=[0, 1]`).

    Example: `target_clusters=[0]`

##### Prefixes for multi-backend models

Some architectures have multiple `mxq` files (e.g. encoder-decoder models). In that case, you can target a specific sub-module by prefixing the parameter name:

- Encoder/decoder prefixes (supported today):
  - `encoder_...` and `decoder_...` variants of the NPU settings above, e.g. `encoder_mxq_path`, `decoder_core_mode`, `encoder_target_cores`, `decoder_target_clusters`, etc.

For example, you can override only the encoder `mxq` file:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    encoder_mxq_path="/path/to/encoder.mxq",
)
```

Other prefixes such as `text_...` / `vision_...` are not currently implemented in this repository.

#### revision

Our quantized models are uploaded on Huggingface Hub.
The repositories on Huggingface Hub work like a git repository, so they can have multiple branches.
We provide multiple quantized variants of a single original model via these branches (revisions).

We use the following revision labels for quantized variants:
- `W8`: all weights are quantized to INT8.
- `W4`: all weights are quantized to INT4.
- `W4V8`: in the attention QKV matrices, the Value (V) matrix is INT8, and the rest are INT4.

Currently, we only upload `W8` and `W4V8` variants.
The default `main` branch may differ by model (some models are fine with `W4V8`, others are not).
If you prefer inference speed over quality, use `W4V8`. If you need higher accuracy, use `W8`.

#### embedding_weight

If you have your own quantized model from our `qbcompiler`, you may have used rotated embedding options.
To make it easier to test custom compiled models, we support overriding the input embedding weights.

- `embedding_weight` (`str`)
    Path to a PyTorch checkpoint file loadable via `torch.load` (commonly `*.pt` or `*.pth`).
    The file can contain:
    - a `torch.Tensor` with shape `[vocab_size, hidden_size]`, or
    - a `dict` (e.g. `state_dict`) containing a `"weight"` entry, or (as a fallback) any single tensor value.

    The tensor must match the model's input-embedding shape exactly; otherwise, loading will fail.
    The weights are copied into `model.get_input_embeddings().weight` (device/dtype are preserved).

#### device

Even though most of the weights are in `mxq` file, some of the layers are not suited for NPU.
For example, embedding layers for LLM models are not computed on NPU, but on CPU.
If you want to change the device for these kinds of CPU layers, you can use `device` parameter.
But, we recommend to use the default value `cpu`, since our `qbruntime` accept inputs from host memory, not from GPU memory.
You should trade off the computation efficiency and memory copy inefficiency.

#### device_map

Same with the `device` parameter, `device_map` sets devices for `cpu` layers.
It doesn't affect our quantized `mxq` models, which can only run on our NPUs.
We recommend not using it.

#### trust_remote_code

For all `transformers` model in our model zoo, you should set `trust_remote_code` to `True` since our auto_map proxy codes are remote code.
With this structure, you can just import `transformers` only, and use auto classes to load our quantized models.
For the sake of performance, our proxy code will only import necessary classes of configs and models.

#### dtype

Same with the `device` parameter, `dtype` sets datatype(eg. `float32`, `bfloat16`) for `cpu` layers.
It doesn't affect our quantized `mxq` models.
We recommend leaving it as the default, since this value is inherited from the original model's `config.json`.

## TPS Benchmark CLI (Sweep)

If you installed the optional extra (`pip install mblt-model-zoo[transformers]`), you can run a simple TPS sweep from the command line:

```bash
mblt-model-zoo tps sweep --model mobilint/Llama-3.2-3B-Instruct --device cpu --revision W8 --json tps.json --plot tps.png
```

Use `--revision` to select a specific quantized revision (e.g., `W8`).
Use `--embedding-weight` to inject custom embedding weights.

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
