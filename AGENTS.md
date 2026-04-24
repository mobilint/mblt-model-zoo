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
  evaluation, datasets, and result types
- `mblt_model_zoo/hf_transformers`: Hugging Face model integrations and utilities
- `mblt_model_zoo/MeloTTS`: MeloTTS integration and text normalization
- `mblt_model_zoo/cli`: CLI entry points and command registration
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
- When adding or renaming exported vision models, update the relevant `__init__.py` files so
  `mblt_model_zoo.vision.list_models()` continues to discover them.

### Tests and Benchmarks

- Many tests depend on Mobilint hardware, downloaded model artifacts, or optional extras.
- Prefer targeted validation over running the entire matrix by default.
- Reuse the documented commands in `tests/vision/TEST.md`, `tests/transformers/TEST.md`,
  `tests/MeloTTS/TEST.md`, and `benchmark/vision/README.md`.
- Use the shared NPU pytest options from `tests/conftest.py` instead of inventing custom flags.

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
pytest tests/transformers/text-generation/test_qwen2.py -k "0.5B"
pytest tests/MeloTTS/test_melo.py -k "KR"
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
