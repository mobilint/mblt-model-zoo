---
name: mblt-transformers
description: Work effectively on Mobilint Model Zoo Hugging Face Transformers integrations and benchmarks.
---

# Mobilint Model Zoo Transformers

Read and follow the canonical skill at
[`../../../.agents/skills/mblt-transformers/SKILL.md`](../../../.agents/skills/mblt-transformers/SKILL.md).

Treat `mblt_model_zoo/cli/tps_table.py` as the source of truth for TPS printed rows and JSON output.
Keep VLM non-batch tests under `image_text_to_text/non_batch`; matrix-runner Phase B owns both batch
text-generation and batch image-text-to-text suites.

Qwen3-VL treats the vision + text MXQ + processor as one release. The top-level
`MobilintQwen3VLConfig.dynamic_vision` bool selects the release: a dynamic-vision release accepts
video and per-prompt multi-image inputs, while a static-vision release supports one image per prompt
only and raises `NotImplementedError` from `MobilintQwen3VLProcessor.__call__` for video or
per-prompt multi-image inputs. Batched single-image prompts are always allowed. The processor reads
the flag from `config.dynamic_vision` in `from_pretrained`. Call
`MobilintQwen3VLProcessor.sync_dynamic_vision_from_model(model)` only when a runtime
`vision_mxq_path=` override diverges from the shipped config. Video decoding requires the
`transformers` extra's `torchcodec` dependency.

EAGLE-3 speculative decoding (`mobilint/EAGLE3-Qwen3-4B` and siblings) loads through
`AutoModelForCausalLM.from_pretrained(...)` as one release bundling base MXQ, one-block draft MXQ,
and FC stack. Tune the draft-tree budget through `GenerationConfig.num_assistant_tokens` (default
`64` in `mblt_model_zoo/hf_transformers/utils/generation_utils.py`); Qwen3-4B measures best around
`25`–`30`. A/B the softmax top-k mode in
`mblt_model_zoo/hf_transformers/utils/eagle3/tree_decoding.py::softmax_topk_cpu_torch`: default
`sliced` renormalizes over the retained top-k slice, and `MBLT_EAGLE3_SOFTMAX_TOPK_MODE=full` (or
`set_softmax_topk_mode(...)`) restores the whole-vocab denominator. The greedy path never enters
that function. `evaluate_posterior` greedy uses `argmax(logits)[safe_positions]` to avoid the
fancy-indexed `(n_cand, depth, vocab)` slice.

`mblt-model-zoo tps measure` on EAGLE-3 pipelines prints the extra rows `accept_steps`,
`tokens_sum`, `tokens_per_step` (= `drafts_avg + 1`, matching `accept_length + 1` in the reference
`eagle3MXQ.py`), and `draft_accept_ratio`; non-EAGLE-3 pipelines omit them. New `tps measure`
options: `--print-output` (diagnostic decoded text), mutually exclusive `--enable-thinking` /
`--disable-thinking` (Qwen3 chat template `<think>` block override), and `--temperature FLOAT`
(`0` = greedy). Chat templates apply to text prompts by default; `tps sweep` stays greedy. On
EAGLE-3, `benchmark_utils.py::_apply_eagle3_gen_kwargs` strips `min_new_tokens` and
`pad_token_id` and sets `eos_token_id=None`, so `--decode N` is an upper bound and measured
`num_decode` reflects actual generation.

MXQ backends exhibit known cross-process non-determinism. Reproduce with
`scripts/probe_mxq_determinism.py`, `scripts/probe_generate_same_process.py`, and
`scripts/probe_warmup_stabilization.py`; warmup does not stabilize across processes. Prefer
same-process `--repeat N` for stable benchmarks.
