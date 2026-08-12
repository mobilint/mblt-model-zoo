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

## EAGLE-3 Workflow

- Load a release (for example `mobilint/EAGLE3-Qwen3-4B`) through
  `AutoModelForCausalLM.from_pretrained(...)`; the wrapper binds the base MXQ, one-block draft
  MXQ, and FC stack as a single release. The presence of `eagle3_base_model` on the loaded model
  is how measurement paths detect EAGLE-3.
- Tune the draft-tree budget through `GenerationConfig.num_assistant_tokens` (default `64` in
  `mblt_model_zoo/hf_transformers/utils/generation_utils.py`). Qwen3-4B measures best in the
  `25`–`30` range: the Hugging Face default of `49` costs more iteration latency than its extra
  acceptance recovers. Override either by editing the shipped `generation_config.json` or by
  setting `model.generation_config.num_assistant_tokens = ...` before `generate`.
- `mblt_model_zoo/hf_transformers/utils/eagle3/tree_decoding.py::softmax_topk_cpu_torch`
  dispatches per call. Default `auto`: slice to the declared `TopKLogitsWarper`'s top-K first
  and apply the processor list on that slice (Hugging Face `_get_logits_warper` order
  Temperature → TopK → TopP makes the slice mathematically identical to full-vocab softmax
  while skipping the full-vocab `exp`); otherwise fall back to the full-vocab path so a bare
  `TopPLogitsWarper` still determines its nucleus over the whole distribution. `full` forces
  the full-vocab path as a manual override. `sliced` is a deprecated back-compat mode that
  unconditionally renormalizes over a top-``max_return_k`` slice and emits a warning; retain
  it only for A/B reproducibility. Toggle at import through
  `MBLT_EAGLE3_SOFTMAX_TOPK_MODE=auto|full|sliced`, or programmatically through
  `set_softmax_topk_mode(...)`. The `max_return_k` argument (default `10`) is a return-slice
  size for downstream candidate matching, not the math slice. Keep `prepare_logits_processor`
  in HF order (RepetitionPenalty → Temperature → TopK → TopP) so the auto slice-by-TopK path
  stays HF-equivalent. The greedy path (`temperature<=1e-5`) never enters this function because
  `prepare_logits_processor` returns `None`.
- Keep the argmax-first shape in `evaluate_posterior` greedy: `argmax(logits)[safe_positions]`
  avoids materializing the `(n_cand, depth, vocab)` fancy-index slice.
- EAGLE-3 speculative-decode rows exposed by `mblt-model-zoo tps measure` are `accept_steps`,
  `tokens_sum`, `tokens_per_step` (= `drafts_avg + 1`, matching `accept_length + 1` in the
  reference `speculative_decoding/mxq_app/eagle3MXQ.py`), and `draft_accept_ratio`. Non-EAGLE-3
  pipelines omit these rows automatically. The schema lives in
  `mblt_model_zoo/cli/tps_table.py`; update it and the focused `tests/transformers/cli_tps`
  suites together.
- `tps measure` also exposes `--print-output` (prints the actually generated tokens for the last
  run, both preserving and stripping special tokens), mutually exclusive
  `--enable-thinking` / `--disable-thinking` (overrides the Qwen3 chat template
  `enable_thinking` flag), and `--temperature FLOAT` (`0.0` keeps greedy; `>0` enables
  `do_sample=True`). Chat templates apply to text prompts by default. `tps sweep` remains greedy
  so its numbers stay comparable.
- On EAGLE-3 pipelines, `_apply_eagle3_gen_kwargs` in
  `mblt_model_zoo/hf_transformers/utils/benchmark_utils.py` strips `min_new_tokens` and
  `pad_token_id` and sets `eos_token_id=None`, so `generate` honors the real EOS from
  `config.json`. `--decode N` becomes an upper bound and the measured `num_decode` reflects the
  tokens actually produced. Non-speculative pipelines keep exact-`N` semantics.
- MXQ backends have known cross-process non-determinism. Reproduce and isolate with
  `scripts/probe_mxq_determinism.py`, `scripts/probe_generate_same_process.py`, and
  `scripts/probe_warmup_stabilization.py`. Warmup does not stabilize outputs across processes
  (verified by the warmup probe). For stable measurements, run same-process `--repeat N`.
- Note the definition drift versus the reference `speculative_decoding/mxq_app` implementation:
  `accept_length` there counts drafts accepted while `tokens_per_step` here counts
  `drafts + 1` (the forced base root token per step). Use `tokens_per_step` when comparing to
  paper-style acceptance numbers.

## Validate Proportionately

- Start with the narrowest test file or documented `-k` selection and use `-x` while iterating.
- Run `pytest tests/transformers --full-matrix` only for release or pre-merge matrix validation.
- Hardware, downloaded models, and external data may be unavailable. Run safe static or focused
  checks and report the limitation rather than broadening the test run.
