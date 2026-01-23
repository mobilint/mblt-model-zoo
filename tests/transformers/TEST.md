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

## Change mxq file

You can change only mxq file with `--mxq-path` params. It can be an absolute path, a relative path from your working directory, or a relative path from local model directory.

```bash
pytest tests/transformers/text-generation/test_llama.py -s -k "Llama-3.2-1B-Instruct" --mxq-path "/path/to/your/Llama-3.2-1B-Instruct.mxq"
```