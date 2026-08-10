---
description: Shared guidance for coding agents working on Mobilint Model Zoo.
paths:
  - "**"
---

# Mobilint Model Zoo Agent Guide

## Scope

`mblt-model-zoo` owns its package integration, Model Zoo CLI, Transformers integrations, and
MeloTTS. `mblt-vision-python` owns all Vision implementation: models, preprocessing,
postprocessing, datasets, evaluation, benchmarks, compilation, Vision tests, and the `mblt-vision`
CLI. Model Zoo retains only a compatibility facade and CLI bridges for its legacy Vision API.

`CLAUDE.md` imports this guide. Keep the shared Model Zoo skill at
`.agents/skills/mblt-model-zoo` and its Claude entry point synchronized. Follow a more-specific
`AGENTS.md` when one exists. User and system instructions take precedence.

Before editing, run `git status --short` and preserve unrelated work.

## Repository Map

- `mblt_model_zoo/cli`: Model Zoo CLI; Vision command handlers are imported from `mblt_vision.cli`.
- `mblt_model_zoo/vision`: compatibility imports and re-exports only; do not restore moved Vision
  implementation here.
- `mblt_model_zoo/compile`: compatibility exports for Vision compilation plus Model Zoo APIs.
- `mblt_model_zoo/hf_transformers`: Hugging Face integrations and benchmark utilities.
- `mblt_model_zoo/MeloTTS`: MeloTTS integration and text normalization.
- `tests`: Model Zoo tests and shared NPU option helpers.

## Engineering Rules

- Read `pyproject.toml`, affected exports, CLI parser, and nearby tests before changing a public
  contract. The package version comes from `mblt_model_zoo.__version__`.
- Use four-space indentation, PEP 484 annotations, Google-style docstrings, and 120-character
  lines. Let Ruff organize imports.
- Catch specific exceptions and provide recovery-oriented errors. Do not catch `Exception` unless
  immediately re-raising or deliberately adding context.
- Preserve local style in `hf_transformers` and `MeloTTS`; they are excluded from repository-wide
  Ruff checks.
- Keep `mblt-model-zoo` CLI help and README examples synchronized. Its Vision subcommands must
  delegate to `mblt_vision.cli`; implement new Vision CLI behavior in `mblt-vision-python` first.
- Keep the Vision facade a thin, documented compatibility layer. Add no new Vision models,
  processing, datasets, evaluation, benchmarks, compilation, or Vision-specific tests here.
- Use `obb` when a Model Zoo compatibility configuration must name the Vision task.

## Transformers and MeloTTS

- Install the matching optional extra before integration tests.
- Keep `mblt-model-zoo tps` table labels, JSON keys, units, and extraction behavior centralized in
  `mblt_model_zoo/cli/tps_table.py`; update its schema and focused tests together.
- Keep non-batch VLM tests under `tests/transformers/image_text_to_text/non_batch`. Run batch
  text-generation and image-text-to-text suites through `scripts/test_transformers_matrix.py`.
- Qwen3-VL dynamic releases accept video and per-prompt multi-image inputs; static releases must
  reject them with the documented `NotImplementedError`. Call
  `MobilintQwen3VLProcessor.sync_dynamic_vision_from_model()` only for an MXQ override that differs
  from the shipped `config.dynamic_vision` setting.

## Validation and Git Safety

- Start with the smallest relevant test or documented `-k` selection. Hardware, downloaded models,
  and external data may be unavailable; report that limitation rather than weakening tests.
- For documentation changes, run `git diff --check` and verify headings and links. For Python
  changes, run focused pytest and `pre-commit run --files <touched files>` when available.
- Do not revert, format, or regenerate unrelated files. Do not add generated artifacts, model
  weights, caches, or benchmark output unless explicitly requested.
