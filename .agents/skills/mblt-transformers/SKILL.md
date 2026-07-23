---
name: mblt-transformers
description: >-
  Work effectively on Mobilint Model Zoo Hugging Face Transformers integrations, tests, and
  benchmarks while respecting optional dependencies, NPU configuration, and download constraints.
---

# Mobilint Model Zoo Transformers

## Start Here

1. Read `AGENTS.md` and the shared `mblt-model-zoo` skill.
2. Run `git status --short` before changing files.
3. Read `pyproject.toml`, `tests/transformers/TEST.md`, and `benchmark/transformers/README.md`.

## Preserve Contracts

- Install the matching `transformers` optional extra before running integration tests.
- Reuse shared NPU options and `tests.npu_backend_options.build_vision_engine_kwargs()` rather
  than introducing divergent hardware flags or engine keyword bundles.
- Preserve local style in `mblt_model_zoo/hf_transformers`; it is excluded from repository-wide
  Ruff checks.

## Validate Proportionately

- Start with the narrowest test file or documented `-k` selection and use `-x` while iterating.
- Run `pytest tests/transformers --full-matrix` only for release or pre-merge matrix validation.
- Hardware, downloaded models, and external data may be unavailable. Run safe static or focused
  checks and report the limitation rather than broadening the test run.
