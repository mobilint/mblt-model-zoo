---
description: Repository-wide instructions for agents working in the Mobilint Model Zoo codebase.
paths:
  - "**"
---

# Agent Guidelines

## Scope

These instructions apply across the repository. For Python implementation details, treat
`mblt_model_zoo/vision`, `tests/vision`, and `benchmark/vision` as the strictest reference area
because they are the most consistently structured.

## Repo Map

- `mblt_model_zoo/vision`: Vision public API, model wrappers, preprocess and postprocess helpers,
  evaluation, datasets, result types, and task subpackages including face detection, OCR, and
  oriented bounding boxes
- `mblt_model_zoo/compile`: Installable compilation APIs, including the optional qbcompiler-backed
  vision pipeline
- `mblt_model_zoo/hf_transformers`: Hugging Face model integrations and utilities
- `mblt_model_zoo/MeloTTS`: MeloTTS integration and text normalization
- `mblt_model_zoo/cli`: CLI entry points and command registration
- `benchmark/transformers`: TPS benchmark scripts and CLI examples for Hugging Face integrations
- `tests`: Feature-specific pytest suites plus shared fixtures in `tests/conftest.py`
- `benchmark`: Dataset organization and benchmark scripts

## Code Style Guide

### Formatting

- Use 4 spaces for indentation.
- Keep lines at 120 characters or fewer.
- Use Ruff for linting, import sorting, and formatting.
- Follow existing local formatting when touching files under `mblt_model_zoo/hf_transformers` or
  `mblt_model_zoo/MeloTTS`, because those paths are excluded from repo-wide Ruff checks in
  `pyproject.toml`.

### Typing and Docstrings

- Use PEP 484 type annotations for all new or modified function and method signatures.
- Use Google-style docstrings for every new or modified module, class, and function.
- Do not duplicate type information or default values in docstrings when the signature already
  communicates them.
- Keep docstrings accurate when behavior changes. Do not leave stale `Args`, `Returns`, or
  `Raises` sections behind.

### Error Handling

- Use `try` and `except` with specific exception types.
- Raise informative errors with clear messages that help the caller recover.
- Avoid catching `Exception` unless you immediately re-raise or intentionally convert the failure
  with added context.

### Imports

- Group imports into standard library, third-party, and local sections.
- Let Ruff and isort-compatible ordering be the source of truth.

## Repo-Specific Python Guidance

### Vision Stack

- Vision model families typically define `ModelInfoSet` enums plus an `MBLT_Engine` subclass.
- Preserve the shape of `model_cfg`, `pre_cfg`, and `post_cfg` objects unless the change
  explicitly updates the public contract.
- Keep public constructor arguments such as `local_path`, `model_type`, `infer_mode`, and
  `product` stable when extending existing models.
- For vision model loading, prefer `model_path` in new docs, examples, and tests. Treat
  `mxq_path` and `onnx_path` as compatibility aliases unless the task specifically targets those
  legacy names.
- Vision framework selection auto-detects `.mxq` and `.onnx` suffixes from `model_path` when
  `framework` is omitted. If an explicit framework conflicts with the local file suffix, preserve
  the fail-fast error behavior instead of silently switching runtimes.
- `mblt_model_zoo.vision` supports both task-subpackage imports and legacy top-level compatibility
  imports such as `from mblt_model_zoo.vision import ResNet50`. For new docs and examples, prefer
  `MBLT_Engine` or task-subpackage imports unless the change is specifically about backward
  compatibility.
- Legacy compatibility wrappers still accept `product`, but the YAML-backed registry ignores it.
  If a change needs non-default artifacts, route that selection through explicit `model_cls`,
  `model_type`, or `model_path` values instead.
- When adding or renaming exported vision models, update the relevant `__init__.py` files so
  `mblt_model_zoo.vision.list_models()` continues to discover them, and keep
  `mblt_model_zoo.vision.__init__` lazy compatibility exports in sync.

### Vision Datasets

- Keep validation dataset definitions in `mblt_model_zoo/vision/datasets/*.yaml`; use the
  Ultralytics-style `path`, `val`, and optional `names` fields, plus this repository's `tasks` and
  `download` metadata.
- Keep dataset class names and category-ID mappings in those YAML files. Preserve the compatibility
  helpers in `mblt_model_zoo/vision/utils/datasets` by loading their values from the registry.
- Use `mblt_model_zoo.vision.datasets.get_dataset_config_for_task()` for task-to-dataset defaults
  instead of duplicating download URLs or cache paths in CLI and benchmark code.
- Preserve the organizer output layouts consumed by the evaluators. In particular, DOTAv1 accepts
  original labels in `labels/val_original` and must retain difficult-object filtering.

### Vision Compilation

- Treat qbcompiler as an optional, compilation-only dependency. Importing `mblt_model_zoo`,
  `mblt_model_zoo.vision`, `mblt_model_zoo.compile.vision`, or the main CLI must continue to work
  when qbcompiler is not installed.
- Keep qbcompiler imports inside the function that starts compilation. Do not add module-level
  qbcompiler imports to packaged compilation or CLI modules.
- A missing qbcompiler installation should fail only when `compile_vision_model()` or
  `mblt-model-zoo compile` is invoked, with a concise installation message. Non-compile CLI
  commands must remain unaffected.
- Keep qbcompiler and the ONNX Runtime dependency needed by the preprocessing engine in the
  `qbcompiler` optional dependency extra rather than the base package dependencies.
- Preserve the three mutually exclusive compilation data levels: `data_path` is the original
  organized image dataset, `subset_path` is an already-sampled image set, and `calib_data_path` is
  a ready directory of preprocessed `.npy` tensors.
- Begin processing at the supplied level. A subset must skip dataset organization and sampling; a
  calibration dataset must skip organization, sampling, and preprocessing and be passed directly
  to qbcompiler after validation.
- Keep compilation defaults independent of the checkout: downloaded and compiled models belong
  under `~/.mblt_model_zoo`, and registry-backed datasets belong under
  `~/.mblt_model_zoo/datasets` unless the user supplies an explicit path.

### Tests and Benchmarks

- Many tests depend on Mobilint hardware, downloaded model artifacts, or optional extras.
- Prefer targeted validation over running the entire matrix by default.
- Reuse the documented commands in `tests/vision/TEST.md`, `tests/transformers/TEST.md`,
  `tests/MeloTTS/TEST.md`, `benchmark/vision/README.md`, and `benchmark/transformers/README.md`.
- The unified vision CLI validation flow currently covers ImageNet-backed image classification,
  COCO-backed object detection, instance segmentation, and pose estimation, WiderFace-backed face
  detection, and DOTAv1-backed oriented bounding boxes.
- `benchmark/vision/README.md` currently documents dataset organization for ImageNet, COCO,
  WiderFace, and DOTAv1. ImageNet, COCO, and DOTAv1 benchmark execution commands are documented
  there today; WiderFace benchmark execution is still pending.
- Use the shared NPU pytest options from `tests/conftest.py` instead of inventing custom flags.
- For vision tests that call `MBLT_Engine(**kwargs)`, prefer the typed helper
  `tests.npu_backend_options.build_vision_engine_kwargs()` so `dev_no`, `core_mode`,
  `target_cores`, and `target_clusters` stay aligned with the engine signature.

#### Transformers Test Scope

Pick the narrowest command that covers the change. `pytest tests/transformers` walks every
category and takes several minutes; only run it when touching shared code such as
`mblt_model_zoo/hf_transformers/utils/generation_utils.py`, base helpers in
`mblt_model_zoo/hf_transformers/utils/`, or other cross-model utilities. Reserve
`pytest tests/transformers --full-matrix` for the pre-merge or release gate, not per-change
iteration. Pass `-x` (stop on first failure) as the default flag while iterating on a fix.

Map the touched source path to a single test file where possible:

- `mblt_model_zoo/hf_transformers/models/qwen2/**` -> `pytest tests/transformers/text_generation/non_batch/test_qwen2.py -k "0.5B"`
- `mblt_model_zoo/hf_transformers/models/qwen3/**` -> `pytest tests/transformers/text_generation/non_batch/test_qwen3.py -k "0.6B"`
- `mblt_model_zoo/hf_transformers/models/llama/**` -> `pytest tests/transformers/text_generation/non_batch/test_llama.py -k "1B"`
- `mblt_model_zoo/hf_transformers/models/exaone/**` -> `pytest tests/transformers/text_generation/non_batch/test_exaone.py -k "2.4B"`
- `mblt_model_zoo/hf_transformers/models/exaone4/**` -> `pytest tests/transformers/text_generation/non_batch/test_exaone4.py`
- `mblt_model_zoo/hf_transformers/models/qwen2_eagle3/**` -> `pytest tests/transformers/text_generation/eagle3/test_qwen2_eagle3.py`
- `mblt_model_zoo/hf_transformers/models/qwen2_vl/**` -> `pytest tests/transformers/image_text_to_text/test_qwen2_vl.py`
- `mblt_model_zoo/hf_transformers/models/qwen3_vl/**` -> `pytest tests/transformers/image_text_to_text/test_qwen3_vl.py -k "2B"`
- `mblt_model_zoo/hf_transformers/models/aya_vision/**` -> `pytest tests/transformers/image_text_to_text/test_aya.py`
- `mblt_model_zoo/hf_transformers/models/blip/**` -> `pytest tests/transformers/image_to_text/test_blip.py`
- `mblt_model_zoo/hf_transformers/models/whisper/**` -> `pytest tests/transformers/automatic_speech_recognition/test_whisper.py -k "small"`
- `mblt_model_zoo/hf_transformers/models/qwen3_asr/**` -> `pytest tests/transformers/automatic_speech_recognition/test_qwen3_asr.py`
- `mblt_model_zoo/hf_transformers/models/bert/**` -> `pytest tests/transformers/fill_mask/test_bert.py -k "bert-base-uncased"`

For model families without a dedicated test file (e.g., `cohere2`, `siglip`) or for edits that
span multiple families in one directory, consult `tests/transformers/TEST.md` for the closest
subdirectory-level scope.

## Comment Style Guide

### Inline Comments

- Use inline comments sparingly.
- Explain why a choice exists, not how the code works.
- Avoid obvious comments that restate the implementation.

### TODOs

- Use `TODO(username): description` for temporary notes or planned follow-ups.

## Validation Guide

Use the smallest validation that meaningfully covers the change.

### Common Setup

```bash
pip install -e . --group dev
```

### Optional Extras

```bash
pip install -e ".[transformers]" --group dev
pip install -e ".[MeloTTS]" --group dev
pip install -e ".[onnxruntime]" --group dev
pip install -e ".[onnxruntime-gpu]" --group dev
pip install -e ".[qwen-asr]" --group dev
pip install -e ".[qbcompiler]" --group dev
```

If validation fails with `ImportError` or `ModuleNotFoundError`, install the relevant optional
extras before retrying.

### Python Checks

```bash
ruff check mblt_model_zoo tests benchmark
ruff format mblt_model_zoo tests benchmark
pre-commit run --files path/to/touched_file.py
```

### Targeted Tests

```bash
pytest tests/vision/test_resnet50.py
pytest tests/transformers/text_generation/non_batch/test_qwen2.py -k "0.5B"
pytest tests/MeloTTS/test_melo.py -k "KR"
pytest tests/vision/test_cli_vision.py
pytest tests/vision/test_wrapper_download.py
```

If required hardware, models, or extras are unavailable, run the narrowest safe validation and
state the limitation clearly.

## Git Rules

### Pre-commit Hook

- Never skip pre-commit hooks. `--no-verify` is forbidden.
- After `.pre-commit-config.yaml` exists, run `pre-commit install` in a fresh clone to register
  the hook.
- If pre-commit fails, fix the underlying issue and rerun it.

### Commits

- Use Conventional Commits such as `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, or `test:`.
- Write the subject in the imperative mood.
- Keep the subject line under 50 characters.
- Keep commits atomic and focused on one logical change.

### Working Tree Safety

- Do not revert unrelated user changes.
- If the repo is already dirty, limit your edits to the files relevant to the task and verify
  diffs carefully before committing.

## Markdown Style Guidelines

### Headers

- Use ATX-style headers.
- Leave a single blank line before and after each header, except for the document title at the
  top.
- Do not skip header levels.

### Lists

- Use hyphens for unordered lists.
- Indent nested list items by exactly 2 spaces when nesting is necessary.
- Use sentence-ending periods only when the list item is a full sentence.

### Code and Links

- Always include a language identifier for fenced code blocks.
- Use inline code for file names, paths, commands, variables, and symbols in prose.
- Prefer descriptive link text and meaningful image alt text.

### Spacing

- Keep exactly one blank line between distinct blocks.
- Avoid trailing whitespace.
- Keep paragraphs concise.

### Frontmatter

- If a Markdown file represents an agent rule, workflow, or reusable operating guide, add YAML
  frontmatter with `description` and `paths`.
