---
name: mblt-model-zoo
description: >-
  Work effectively in the Mobilint Model Zoo repository. Use for changes to vision models and
  datasets, optional qbcompiler compilation, Hugging Face Transformers integrations, MeloTTS, CLI
  commands, tests, benchmarks, or repository documentation that must respect Mobilint NPU and
  model-download constraints.
---

# Mobilint Model Zoo

## Start Here

1. Read `AGENTS.md`; it is the canonical shared agent guide and `CLAUDE.md` imports it.
2. Run `git status --short` before changing files.
3. Read `pyproject.toml` and the nearest area guide before choosing dependencies or validation:
   - Vision: `tests/vision/TEST.md` and `benchmark/vision/README.md`.
   - Transformers: `tests/transformers/TEST.md` and `benchmark/transformers/README.md`.
   - MeloTTS: `tests/MeloTTS/TEST.md` and `mblt_model_zoo/MeloTTS/README.md`.
   - Compilation: `compile/vision/README.md`.

## Select the Correct Surface

- Use `mblt_model_zoo.vision.MBLT_Engine` for new vision loading code. Prefer `model_path`; treat
  `mxq_path` and `onnx_path` as compatibility aliases.
- Update task package exports and the lazy top-level vision exports together when adding or
  renaming a vision model. Confirm `list_models()` discovery.
- Use the YAML dataset registry and `get_dataset_config_for_task()` for vision dataset defaults.
- Keep qbcompiler optional and lazily imported. Use `compile_vision_model()` or the `compile` CLI
  only after the `qbcompiler` extra is installed.
- Use the installed `mblt-model-zoo` CLI for package behavior. Its native commands are `predict`,
  `val`, `compile`, `tps`, `melo`, and `melo-ui`; `classify`, `detect`, `pose`, `segment`, and
  `melotts` are aliases.

## Preserve Important Contracts

- Keep legacy vision constructor arguments and imports working unless the task explicitly changes
  compatibility.
- Preserve automatic `.mxq`/`.onnx` framework detection and errors for conflicting explicit
  framework selections.
- Preserve the compile entry-level rules: `data_path`, `subset_path`, and `calib_data_path` are
  mutually exclusive and each skips prior preparation stages.
- Use an explicit default seed of `0` for new vision randomness.
- When a package update changes a durable public fact or workflow, update `AGENTS.md` and this
  skill in the same change. Keep the Claude entry point thin; shared content belongs here.
- Do not force formatting standards on `hf_transformers` or `MeloTTS`; follow local style.

## Validate Proportionately

- Install only the extra needed for the touched area.
- Run one targeted test file or `-k` selection first; append `-x` while fixing failures.
- Expect NPU tests and model-download tests to be environment-dependent. Report unavailable
  hardware, artifacts, or extras rather than broadening the test run.
- Run `pre-commit run --files <touched files>` for Python edits when hooks are available. For docs,
  run `git diff --check` and inspect rendered Markdown structure.

## Avoid

- Do not add module-level qbcompiler imports or turn optional dependencies into base dependencies.
- Do not duplicate dataset URLs, paths, or long test commands that a registry or local guide owns.
- Do not revert unrelated user work, commit generated outputs, or bypass pre-commit hooks.
