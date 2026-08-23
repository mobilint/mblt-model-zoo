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

`MobilintNPUBackend` hosts `N` `qbruntime.Model` slots; `max_batch_size` is the aggregate `N*K`
capacity with `K` = compiled MXQ batch axis. `N = ceil(max_batch_size / K)` slots are launched
round-robin across the unique devices referenced by the canonical target strings. Non-batch MXQ
(`K=1`) with `B>1` fans into `N=B` slots dispatched in parallel via
`MobilintNPUBackend.infer_slot`. Beam search stays `N=1`. `dev_no` is syntactic sugar for the
device-prefix component of the canonical target strings — scalar pins one device, list expands
to many. Read the canonical `target_cores` (`"d:c:k"`) / `target_clusters` (`"d:c"`) accessors
at dispatch time, not `dev_no`. Backend target topology accumulates raw overrides in
`NPUTargetSpecPending` on `MobilintNPUBackend._pending`; the lazy `_spec` property calls
`pending.finalize()` once per epoch and caches on `self._finalized`, then promotes `_pending`
to a fresh baseline via `NPUTargetSpecPending.from_baseline` so the next setter chain (or
standalone runtime mutation) gets a clean intent slate. Setters (`dev_no`, `core_mode`,
`target_cores`, `target_clusters`) only mutate `_pending` and invalidate `_finalized`; setter
order within one chain does not affect the resolved spec. `finalize()` runs one
ordered pipeline (legacy migration → sibling drop → grain unification → off-mode drop →
device-set consistency → `global8` coverage). Target-only override syncs `dev_no` to the
target device set; `dev_no`-only override clears stale targets and re-expands sugar; both
overridden → device-set consistency check surfaces mismatches on the next canonical read (not
on the setter). `NPUTargetSpec.from_kwargs` remains the config-layer (JSON load) entry point.
Legacy 2-part cores, bare int clusters, and `qbruntime.CoreId`/`Cluster` objects are silently
migrated to canonical form inside `finalize()` (and its `_normalize_npu_target_kwargs`
config-layer wrapper); `single` unfolds clusters to cores, `multi`/`global4`/`global8` fold
cores to clusters and warn on partial coverage, `global8` requires both clusters on every
covered device. `MobilintCache([m0, m1, ...], per_model_batch=K)` dualizes KV along
`(model_idx, cache_id)` — row `i` maps to `(i//K, i%K)`; `slot_of` / `model_of` /
`group_by_model` expose the routing. `MobilintBeamCache` enforces `N=1`. On HBM `BadAlloc`,
`create`/`launch` disposes every previously loaded slot and re-raises as
`MobilintBackendAllocError` with phase/slot/dev/counts context — lower `max_batch_size` or
spread across more devices via `dev_no`.

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
and FC stack. Qwen3 and Llama base families live under
`mblt_model_zoo/hf_transformers/models/{qwen3_eagle3,llama_eagle3}/`
(`MobilintQwen3Eagle3ForCausalLM`, `MobilintLlamaEagle3ForCausalLM`) and share
`MobilintEagle3BaseModelMixin` / `MobilintEagle3DraftModelMixin`; embed_tokens + rotary_emb init
lives in the mixins so every concrete `MobilintXxxEagle3ForCausalLM.__init__` is a thin wiring
shim (subclass the mixins to register a new base family). Tune the draft-tree budget through
`GenerationConfig.num_assistant_tokens` (default `64` in
`mblt_model_zoo/hf_transformers/utils/generation_utils.py`); Qwen3-4B measures best around
`25`–`30`. The softmax dispatch in
`mblt_model_zoo/hf_transformers/utils/eagle3/tree_decoding.py::softmax_topk_cpu_torch` defaults
to `auto`: slice by declared `TopKLogitsWarper` if present, else fall back to full-vocab so a
bare `TopPLogitsWarper` still determines its nucleus over the whole distribution. The
slice-by-TopK path detects boundary ties (HF's strict-less-than `TopKLogitsWarper` keeps every
logit equal to the k-th threshold; `torch.topk` drops tied entries at the boundary) via
`(x >= threshold).sum(-1) > slice_size` and falls back to the full-vocab helper when true, so
it stays HF-equivalent even under ties. `full` is a manual full-vocab override; `sliced` is a
deprecated legacy top-``max_return_k`` renormalization that violates HF nucleus semantics
without a TopK companion and emits a warning. Toggle via
`MBLT_EAGLE3_SOFTMAX_TOPK_MODE=auto|full|sliced` or `set_softmax_topk_mode(...)`. Keep
`prepare_logits_processor` in HF order (RepetitionPenalty → Temperature → TopK → TopP) so the
auto slice-by-TopK path stays mathematically equivalent to full-vocab (modulo the tie
fallback). The greedy path never enters that function. `evaluate_posterior` greedy uses `argmax(logits)[safe_positions]` to
avoid the fancy-indexed `(n_cand, depth, vocab)` slice.

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
