---
name: mblt-model-zoo
description: Use this skill when working in the Mobilint Model Zoo repository. It covers the repo layout, common edit patterns for vision model wrappers, transformers and MeloTTS extras, CLI entry points, and safe validation strategies when Mobilint NPU hardware or downloaded models may be unavailable.
paths:
  - "**"
---

# Mobilint Model Zoo

## When to Use This Skill

Use this skill for changes anywhere in this repository, especially when the task involves:

- Vision model wrappers, preprocessors, postprocessors, datasets, or evaluation helpers
- Hugging Face transformer integrations under `mblt_model_zoo/hf_transformers`
- MeloTTS integration under `mblt_model_zoo/MeloTTS`
- CLI entry points under `mblt_model_zoo/cli`
- Pytest or benchmark changes
- Repo docs that need to reflect the project's hardware-dependent workflows

## First Reads

Start with the smallest set of files that anchor the task:

- `pyproject.toml` for dependencies, extras, scripts, Ruff settings, and Python versions
- `README.md` for the public package contract
- Area-specific docs such as `tests/vision/TEST.md`, `tests/transformers/TEST.md`,
  `tests/MeloTTS/TEST.md`, `benchmark/vision/README.md`, or
  `mblt_model_zoo/hf_transformers/README.md`
- `git status --short` before editing so you do not overwrite unrelated user changes

## Repo Map

- `mblt_model_zoo/vision`: Vision public API, model wrappers, preprocessing, postprocessing,
  datasets, evaluation, and result objects
- `mblt_model_zoo/hf_transformers`: Custom model, config, and proxy integrations for Transformers
- `mblt_model_zoo/MeloTTS`: MeloTTS runtime, API, CLI glue, and text normalization utilities
- `mblt_model_zoo/cli`: Installed CLI entry points. `mblt-model-zoo` is defined in
  `pyproject.toml`
- `tests`: Pytest suites grouped by feature area
- `benchmark`: Dataset organization and benchmark scripts

## Working Conventions

### Vision Model Work

Most vision model files follow the same pattern:

1. One or more `ModelInfoSet` enums define `model_cfg`, `pre_cfg`, and `post_cfg`.
2. An `MBLT_Engine` subclass exposes the public model constructor.
3. The model family is re-exported through the task package `__init__.py`, then through
   `mblt_model_zoo/vision/__init__.py`.

When editing these files:

- Keep `OrderedDict`-based config blocks unless the surrounding file already moved away from that
  pattern.
- Preserve public constructor arguments such as `local_path`, `model_type`, `infer_mode`, and
  `product`.
- Be careful with changes that affect `list_models()` discovery. Exported classes must remain
  subclasses of `MBLT_Engine`.

### Shared Vision Pipeline Work

For preprocess or postprocess changes, check these modules first:

- `mblt_model_zoo/vision/wrapper.py`
- `mblt_model_zoo/vision/utils/preprocess`
- `mblt_model_zoo/vision/utils/postprocess`
- `mblt_model_zoo/vision/utils/results.py`
- `mblt_model_zoo/vision/utils/types.py`

Preserve behavior for both local `.mxq` paths and Hugging Face downloads unless the task
explicitly changes that contract.

### Transformers and MeloTTS Work

These areas have optional dependencies and heavier runtime assumptions.

- Install the matching extra before test execution.
- Prefer the commands documented in the corresponding `tests/*/TEST.md`.
- Narrow test scope with a subdirectory, file path, or `-k` filter before attempting a full suite.
- Expect some tests to require Mobilint hardware, model downloads, or extra data files.
- `tool.ruff.exclude` currently excludes `mblt_model_zoo/hf_transformers` and
  `mblt_model_zoo/MeloTTS`, so preserve local style there instead of forcing broad cleanup.

### Documentation Work

- Follow the Markdown rules in `AGENTS.md`.
- Keep docs operational and repo-specific.
- For rule or workflow documents, include YAML frontmatter with at least `description` and
  `paths`.

## Validation

Use the smallest validation that meaningfully covers your change.

### Base Environment

```bash
pip install -e . --group dev
```

### Optional Extras

```bash
pip install -e ".[transformers]" --group dev
pip install -e ".[MeloTTS]" --group dev
```

### Lint and Format Python

```bash
ruff check mblt_model_zoo tests benchmark
ruff format mblt_model_zoo tests benchmark
pre-commit run --files path/to/touched_file.py
```

### Targeted Tests

```bash
pytest tests/vision/test_resnet50.py
pytest tests/transformers/text-generation/test_qwen2.py -k "0.5B"
pytest tests/MeloTTS/test_melo.py -k "KR"
```

### Hardware-Aware Sweeps

Transformers tests expose shared NPU options through `tests/conftest.py`. Reuse existing flags
such as:

```text
--core-mode
--vision-core-mode
--text-core-mode
--encoder-core-mode
--decoder-core-mode
```

If the environment does not provide Mobilint NPU runtime support, stop at static checks or the
narrowest safe test and call out the limitation explicitly.

## Things to Avoid

- Do not rewrite large generated or vendor-like sections just to normalize style.
- Do not broaden test scope by default. Full suites are expensive and often hardware-bound.
- Do not revert unrelated work already present in the tree.
- Do not duplicate long command instructions across docs when an existing README or `TEST.md`
  already owns them.
