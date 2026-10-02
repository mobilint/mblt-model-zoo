---
description: Shared guidance for coding agents working on Mobilint Model Zoo.
paths:
  - "**"
---

# Mobilint Model Zoo Agent Guide

## Scope

`mblt-model-zoo` owns its package integration and the Model Zoo CLI. `mblt-vision-python`
owns all Vision implementation: models, preprocessing, postprocessing, datasets, evaluation,
benchmarks, compilation, Vision tests, and the `mblt-vision` CLI. `transformers-mblt` owns all
Hugging Face Transformers implementation: models, caches, generation, EAGLE-3, the Hub proxies, the
TPS and upstream-passthrough CLI, benchmarks, scripts, and Transformers tests. `melotts-mblt` owns all
MeloTTS implementation: the TTS API, synthesizer, text processing, the `melotts-mblt` CLI, and MeloTTS tests.
Model Zoo retains only compatibility facades and CLI bridges for its legacy Vision, Transformers, and MeloTTS APIs.

`CLAUDE.md` imports this guide. The Claude entry point at `.claude/skills/mblt-model-zoo/SKILL.md`
is a symlink to its `.agents/skills/...` counterpart, so editing the `.agents` copy is enough.
Follow a more-specific `AGENTS.md` when one exists. User and system instructions take precedence.

Before editing, run `git status --short` and preserve unrelated work.

## Repository Map

- `mblt_model_zoo/cli`: Model Zoo CLI. Vision command handlers are imported from `mblt_vision.cli`;
  `tps.py`, `tps_table.py`, `chat.py`, and `transformers_compat.py` are aliases of their
  `transformers_mblt.cli` modules, with install-hint fallbacks in `_transformers.py` when the
  `transformers` extra is missing.
- `mblt_model_zoo/vision`: compatibility imports and re-exports only. Every compatibility module
  must forward to `mblt_vision`; do not restore copied Vision implementation, model YAMLs, or
  dataset YAMLs here. Task submodules (`mblt_model_zoo.vision.<task>`) are registered dynamically
  in `vision/__init__.py` from `mblt_vision.list_tasks()`; do not add a physical per-task stub
  package here — a new mblt_vision task becomes importable with no Model Zoo change.
- `mblt_model_zoo/compile`: compatibility exports for Vision compilation plus Model Zoo APIs.
- `mblt_model_zoo/hf_transformers`: forwarding-only facade. Its `__init__.py` installs a meta-path
  alias so every `mblt_model_zoo.hf_transformers.<path>` import is the same module object as
  `transformers_mblt.<path>`. Existing user code and Hub revisions pinned before the proxy update
  (whose `proxy_*.py` imports only the legacy path) depend on this; current Hub proxies import
  `transformers_mblt` first. Do not add copied Transformers implementation, per-module stub files,
  tests, or benchmarks here.
- `mblt_model_zoo/MeloTTS`: forwarding-only facade over `melotts_mblt`, using the same shared alias
  (`mblt_model_zoo/_standalone_alias.py`) as `hf_transformers`. Do not add copied MeloTTS implementation, text
  data, tests, or benchmarks here.
- `tests`: Model Zoo tests and shared NPU option helpers.

## Engineering Rules

- Read `pyproject.toml`, affected exports, CLI parser, and nearby tests before changing a public
  contract. The package version comes from `mblt_model_zoo.__version__`.
- Use four-space indentation, PEP 484 annotations, Google-style docstrings, and 120-character
  lines. Let Ruff organize imports.
- Catch specific exceptions and provide recovery-oriented errors. Do not catch `Exception` unless
  immediately re-raising or deliberately adding context.
- Keep `mblt-model-zoo` CLI help and README examples synchronized. Its Vision subcommands must
  delegate to `mblt_vision.cli`; implement new Vision CLI behavior in `mblt-vision-python` first.
- Pass board selection through to the standalone Vision/NPU packages with normalized
  `target_device` values. Do not restore legacy Vision artifact lookup or product-specific backend
  code in Model Zoo.
- Supported target-device identifiers are `aries-rb`, `regulus-ra`, `regulus-rb`,
  `regulus-ra-usb`, and `regulus-rb-usb`; require `mblt-npu-python>=0.1.0` (which pulls
  `mobilint-qb-runtime>=1.4.0`). The Transformers board-setter contract (every `*target_device`
  setter on `MobilintConfigMixin` and the multi-backend mixins routes through
  `_rebuild_backend_for_target_device`) is implemented and tested in transformers-mblt.
- `core_mode="auto"` is the default fallback when a model config does not specify a mode. It is
  supported for MXQs compiled with `qbcompiler>=1.3.0` and requires `mobilint-qb-runtime>=1.4.0`;
  Batch LLM artifacts may select `single`, `global4`, or `global8` per layer at runtime.
- Keep the Vision facade a thin, documented compatibility layer. Add no new Vision models,
  processing, datasets, evaluation, benchmarks, or compilation here. The only Vision tests kept
  here are generic, opt-in facade smoke tests under `tests/vision`; implementation-specific tests
  belong in `mblt-vision-python`.
- Build release artifacts from a clean tree and inspect the wheel contents. Model Zoo distributions
  must not contain legacy Vision `models/`, `utils/preprocess/`, `utils/postprocess/`,
  `utils/datasets/`, or `utils/evaluation/` paths; those belong exclusively to
  `mblt-vision-python`. `mblt_model_zoo/hf_transformers/` and `mblt_model_zoo/MeloTTS/` must contain only their
  facade `__init__.py`; their implementations belong exclusively to `transformers-mblt` and `melotts-mblt`.
- Use `obb` when a Model Zoo compatibility configuration must name the Vision task.

## Transformers and MeloTTS

- Implement Transformers behavior (models, caches, generation, EAGLE-3, TPS schema, CLI options, Hub
  proxies) in `transformers-mblt` first; Model Zoo picks it up through the facade and CLI bridges.
  Its `AGENTS.md` and `transformers-mblt` skill hold the NPU backend, cache, Qwen3-VL, and EAGLE-3
  contracts.
- The `transformers` extra installs `transformers-mblt`; `qwen-asr` installs `transformers-mblt[qwen-asr]`;
  `MeloTTS` installs `melotts-mblt`. The base package must keep importing, and its other CLI commands must keep
  working, without these extras.
- Keep `tests/transformers` limited to facade and bridge compatibility tests
  (`test_transformers_facade.py`); functional, contract, and NPU Transformers tests belong in
  transformers-mblt.
- Implement MeloTTS behavior (TTS API, synthesizer, text processing, CLI options) in `melotts-mblt` first; its
  `AGENTS.md` and `melotts-mblt` skill hold the contracts. `cli/melo.py`, `cli/melo_ui.py`, and
  `utils/melotts_download.py` bridge to `melotts_mblt.cli` (`run_tts`, `run_ui`, `run_download`) with
  install-hint fallbacks in `cli/_melotts.py`. Keep `tests/MeloTTS` limited to facade and bridge compatibility tests
  (`test_melotts_facade.py`).
- `utils/npu_backend.py` keeps the historical `MobilintNPUBackend.dispatcher` property for Model Zoo
  callers and builds the transformers-mblt `MultiSlotDispatcher` (shared `_mblt_model_zoo_dispatcher`
  attribute), so both packages can install it without clobbering each other.

## Validation and Git Safety

- Start with the smallest relevant test or documented `-k` selection. Hardware, downloaded models,
  and external data may be unavailable; report that limitation rather than weakening tests.
- For documentation changes, run `git diff --check` and verify headings and links. For Python
  changes, run focused pytest and `pre-commit run --files <touched files>` when available.
- Do not revert, format, or regenerate unrelated files. Do not add generated artifacts, model
  weights, caches, or benchmark output unless explicitly requested.
- When a significant package change lands (public API, CLI bridge, dependency/runtime, or ownership
  boundary), update this guide, `.agents/skills/mblt-model-zoo/SKILL.md` (its `.claude` entry point
  is a symlink, so no separate update is needed there), and the relevant README in the same change.
