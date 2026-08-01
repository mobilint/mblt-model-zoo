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
- Treat `mblt_model_zoo/cli/tps_table.py` as the source of truth for TPS printed rows, JSON keys,
  units, and run/aggregate/summary extraction. Update the focused `tests/transformers/cli_tps`
  schema and layer-consistency tests with any change.
- Keep VLM non-batch tests under `tests/transformers/image_text_to_text/non_batch`. Keep batch
  text-generation and image-text-to-text suites in their `batch` directories and route both
  through serial Phase B in `scripts/test_transformers_matrix.py`.
- Qwen3-VL release contract: `MobilintQwen3VLConfig.dynamic_vision` (top-level bool) pairs the
  vision MXQ, text MXQ, and processor as one release. A dynamic-vision release accepts video and
  per-prompt multi-image inputs. A static-vision release supports one image per prompt only and
  raises `NotImplementedError` from `MobilintQwen3VLProcessor.__call__` for video or per-prompt
  multi-image, with a message pointing the caller at a dynamic-vision release. Batched
  single-image prompts are always allowed. `MobilintQwen3VLProcessor.from_pretrained` derives
  the flag from `config.dynamic_vision` and syncs the video processor's mirror. Call
  `MobilintQwen3VLProcessor.sync_dynamic_vision_from_model(model)` only when a runtime
  `vision_mxq_path=` override diverges from the shipped config so the processor adopts the
  detected `visual._uses_dynamic_vision` value.
- Qwen3-VL video decoding requires the `torchcodec` dependency shipped with the `transformers`
  extra; validate video paths only against a dynamic-vision release.
- Preserve local style in `mblt_model_zoo/hf_transformers`; it is excluded from repository-wide
  Ruff checks.

## Validate Proportionately

- Start with the narrowest test file or documented `-k` selection and use `-x` while iterating.
- Run `pytest tests/transformers --full-matrix` only for release or pre-merge matrix validation.
- Hardware, downloaded models, and external data may be unavailable. Run safe static or focused
  checks and report the limitation rather than broadening the test run.
