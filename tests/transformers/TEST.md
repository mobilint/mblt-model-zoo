# Test `transformers`

You can validate Mobilint's Transformers integration with [`pytest`](https://docs.pytest.org/en/stable/). The snippets below assume your virtual environment is already activated.

## Install Development Dependencies

Install the runtime extras plus the developer tooling (pytest, datasets, torchvision, audio libs, etc.) required by the test suite:

```bash
pip install -e ".[transformers]" --group dev
```

## Run All Tests

Execute the entire Transformers test matrix (this may take a while because it loads every supported model):

```bash
pytest tests/transformers
```

## Run a Subdirectory of Tests

Limit execution to a test category, e.g., only causal language-model tests:

```bash
pytest tests/transformers/text-generation
```

## Run a Single Test File

Target a specific file to focus on one model family:

```bash
pytest tests/transformers/image-text-to-text/test_aya.py
```

## Run a Single Model Case

Many tests are parameterized over multiple `mobilint/*` model IDs. Use `-k` to run just one of those cases, e.g., the smallest Qwen variant inside the causal LM suite:

```bash
pytest tests/transformers/text-generation/test_qwen2.py -k "Qwen2.5-0.5B-Instruct"
```

Or you can just write the part of the model name.

```bash
pytest tests/transformers/text-generation/test_qwen2.py -k "0.5B"
```

## Keyword Parameters

For any test, you can use keyword parameters explained in README.md [Keyword Parameters](../../mblt_model_zoo/hf_transformers/README.md#keyword-parameters) section. Please note that overriding parameters affect every tests you run. We recommend to narrow tests down to only single model case, described [above](#run-a-single-model-case)

```
  --mxq-path=MXQ_PATH   Override default mxq_path for pipeline loading.
  --dev-no=DEV_NO       NPU device number.
  --core-mode=CORE_MODE
                        NPU core mode (single, multi, global4, global8).
  --target-cores=TARGET_CORES
                        Target cores (e.g., "0:0;0:1;0:2;0:3").
  --target-clusters=TARGET_CLUSTERS
                        Target clusters (e.g., "0;1").
  --encoder-mxq-path=ENCODER_MXQ_PATH
                        Override encoder mxq_path.
  --encoder-dev-no=ENCODER_DEV_NO
                        encoder NPU device number.
  --encoder-core-mode=ENCODER_CORE_MODE
                        encoder NPU core mode (single, multi, global4, global8).
  --encoder-target-cores=ENCODER_TARGET_CORES
                        encoder target cores (e.g., "0:0;0:1;0:2;0:3").
  --encoder-target-clusters=ENCODER_TARGET_CLUSTERS
                        encoder target clusters (e.g., "0;1").
  --decoder-mxq-path=DECODER_MXQ_PATH
                        Override decoder mxq_path.
  --decoder-dev-no=DECODER_DEV_NO
                        decoder NPU device number.
  --decoder-core-mode=DECODER_CORE_MODE
                        decoder NPU core mode (single, multi, global4, global8).
  --decoder-target-cores=DECODER_TARGET_CORES
                        decoder target cores (e.g., "0:0;0:1;0:2;0:3").
  --decoder-target-clusters=DECODER_TARGET_CLUSTERS
                        decoder target clusters (e.g., "0;1").
  --vision-mxq-path=VISION_MXQ_PATH
                        Override vision mxq_path.
  --vision-dev-no=VISION_DEV_NO
                        vision NPU device number.
  --vision-core-mode=VISION_CORE_MODE
                        vision NPU core mode (single, multi, global4, global8).
  --vision-target-cores=VISION_TARGET_CORES
                        vision target cores (e.g., "0:0;0:1;0:2;0:3").
  --vision-target-clusters=VISION_TARGET_CLUSTERS
                        vision target clusters (e.g., "0;1").
  --text-mxq-path=TEXT_MXQ_PATH
                        Override text mxq_path.
  --text-dev-no=TEXT_DEV_NO
                        text NPU device number.
  --text-core-mode=TEXT_CORE_MODE
                        text NPU core mode (single, multi, global4, global8).
  --text-target-cores=TEXT_TARGET_CORES
                        text target cores (e.g., "0:0;0:1;0:2;0:3").
  --text-target-clusters=TEXT_TARGET_CLUSTERS
                        text target clusters (e.g., "0;1").
  --revision=REVISION   Override model revision (e.g., W8).
  --embedding-weight=EMBEDDING_WEIGHT
                        Path to custom embedding weights.
```