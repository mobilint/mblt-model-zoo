---
name: mblt-model-zoo
description: >-
  Work on the Mobilint Model Zoo package, its legacy Vision, Transformers, and MeloTTS compatibility
  facades, CLI integration, and repository documentation. Use for Model Zoo changes; implement Vision
  features in mblt-vision-python, Transformers features in transformers-mblt, and MeloTTS features in
  melotts-mblt instead.
---

# Mobilint Model Zoo

`mblt-model-zoo` is deprecated since 2.13.0 and receives no feature updates. Accept only deprecation,
compatibility, packaging, and critical fixes; route new work to mblt-vision-python, transformers-mblt,
melotts-mblt, or mblt-npu-python. Keep the notices in `mblt_model_zoo/_deprecation.py` accurate (see the
`AGENTS.md` Deprecation Contract), keep `mblt_model_zoo/__init__.py` lazy, and keep every facade working.

1. Read `AGENTS.md`, run `git status --short`, and inspect the relevant parser, exports, tests, and
   `pyproject.toml` before editing.
2. Keep `mblt_model_zoo.vision` and its Vision compilation exports as thin compatibility layers.
   Compatibility modules must forward to `mblt_vision`; never copy Vision implementation, model
   YAMLs, dataset YAMLs, evaluation, or benchmarks into Model Zoo. Limit
   `tests/vision` to generic opt-in facade smoke tests; keep implementation-specific Vision tests
   in `mblt-vision-python`.
3. Make new Vision CLI behavior in `mblt-vision-python` first. The Model Zoo `predict`, `val`, and
   `compile` handlers must delegate to `mblt_vision.cli`.
4. Preserve Model Zoo CLI help and README examples when its CLI integration changes. Pass
   board-specific `target_device` through to the standalone packages; do not reintroduce legacy
   product/artifact selection in Model Zoo. Supported boards are `aries-rb`, `regulus-ra`,
   `regulus-rb`, `regulus-ra-usb`, and `regulus-rb-usb`; the shared runtime floor is
   `mblt-npu-python>=0.1.0` / `mobilint-qb-runtime>=1.4.0`. The Transformers `*target_device`
   setter contract (routing through `_rebuild_backend_for_target_device`) lives in transformers-mblt.
   `core_mode="auto"` is the default fallback when model config omits a mode. It requires MXQs
   compiled with `qbcompiler>=1.3.0` and `mobilint-qb-runtime>=1.4.0`, and lets Batch LLM
   artifacts choose `single`, `global4`, or `global8` per layer.
5. Keep `mblt_model_zoo.hf_transformers` a forwarding-only facade: its meta-path alias makes every
   `mblt_model_zoo.hf_transformers.<path>` import the same module object as `transformers_mblt.<path>`,
   which existing user code and Hub revisions pinned before the proxy update rely on (current Hub
   proxies import `transformers_mblt` first). `cli/tps.py`, `tps_table.py`, `chat.py`, and
   `transformers_compat.py` alias `transformers_mblt.cli`; keep their install-hint fallbacks so the
   base package and non-Transformers commands work without the `transformers` extra. Implement
   Transformers models, TPS schema, EAGLE-3, and Qwen3-VL contracts in transformers-mblt (its
   `transformers-mblt` skill), and keep `tests/transformers` to facade/bridge compatibility tests.
   Keep `mblt_model_zoo.MeloTTS` a forwarding-only facade over `melotts_mblt` through the same
   `_standalone_alias.install_alias`; `cli/melo.py`, `cli/melo_ui.py`, and `utils/melotts_download.py` bridge to
   `melotts_mblt.cli` with install-hint fallbacks, and `tests/MeloTTS` holds only facade/bridge tests.
6. Start with focused tests. Report unavailable hardware, downloads, and optional extras instead of
   weakening validation. For docs, run `git diff --check`.
7. When package ownership, public API, CLI bridges, or runtime dependencies change significantly,
   update `AGENTS.md`, this canonical skill, the Claude entry point when its workflow changes, and
   relevant documentation in the same change.
8. Before a release, build from a clean tree and inspect the wheel. Reject legacy Vision
   implementation paths such as `vision/models/`, `vision/utils/preprocess/`,
   `vision/utils/postprocess/`, `vision/utils/datasets/`, and `vision/utils/evaluation/`, and any
   `hf_transformers/models/` or `hf_transformers/utils/` path, and any `MeloTTS/` file other than `__init__.py`.
