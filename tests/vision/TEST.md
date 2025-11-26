# Test `vision`

You can validate Mobilint's Vision API with [`pytest`](https://docs.pytest.org/en/stable/). The snippets below assume your virtual environment is already activated.

## Install Packages

Install the runtime extras plus the developer tooling required by the test suite:

```bash
pip install -e .
pip install pytest
```

## Run All Tests

Execute the entire Vision test matrix (this may take a while because it loads every supported model):

```bash
pytest tests/vision
```

## Run a Single Test File

Target a specific test file to focus on one model family:

```bash
pytest tests/vision/test_resnet50.py
```
