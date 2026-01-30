# Test `MeloTTS`

You can validate Mobilint's MeloTTS integration with [`pytest`](https://docs.pytest.org/en/stable/). The snippets below assume your virtual environment is already activated.

## Install Development Dependencies

Install the runtime extras plus the developer tooling (pytest, librosa, etc.) required by the test suite:

```bash
pip install -e ".[MeloTTS]" --group dev
```

## Run All Tests

Execute the entire Transformers test matrix (this may take a while because it loads every supported model):

```bash
pytest tests/MeloTTS
```

## Run a Single Language Case

Use `-k` to run just one of the languages, e.g., tests only Korean:

```bash
pytest tests/MeloTTS/test_melo.py -k "KR"
```
