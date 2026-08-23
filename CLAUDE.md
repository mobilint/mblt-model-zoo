# Claude Code Guide

@AGENTS.md

## Claude-Specific Notes

- Treat `AGENTS.md` as the canonical repository guidance. When a durable workflow changes, update
  it alongside this guide and both Codex and Claude skills.
- Keep this guide synchronized with `AGENTS.md` for shared repository guidance so Claude Code and
  Codex receive the same workflow requirements.
- The Claude Code shared-skill entry point is `.claude/skills/mblt-model-zoo/SKILL.md`; its shared
  content is maintained in `.agents/skills/mblt-model-zoo/SKILL.md`. Use the focused Vision and
  Transformers entries in `.claude/skills/mblt-vision/` and `.claude/skills/mblt-transformers/`.
- When the CLI changes, keep parser `-h`/`--help` text and the README CLI guide synchronized; check
  the root help and each affected subcommand help output.
- NYU Depth organization installs only its 654 validation image/depth pairs as `images/` and `depth/` at the output root.
- NYU Depth evaluation uses stretched inputs, per-image median alignment, and pooled target-valid-pixel statistics;
  it rejects non-finite predictions at those pixels and reports `delta1`, `abs_rel`, and `rmse`.
- `eval_imagenet()` returns Top-1 as a float for compatibility; `eval_imagenet_metrics()` exposes
  structured Top-1 primary and Top-5 secondary metrics.
- Evaluator result objects expose `primary_score` and `secondary_score`; `eval_coco()` remains
  numeric while `eval_coco_metrics()` exposes structured mAP50-95 and mAP50.
- The standard Vision benchmark runner records NYU depth metrics and dispatches ADE20K or
  Cityscapes semantic metrics from `post_cfg.dataset`.
- Auto-detect `.onnx` benchmark `--model-path` values before core-mode expansion and record one
  neutral `onnx` target for `--core-mode all`.
- DOTAv1 validation uses rotated mAP50-95 as its primary metric and rotated mAP50 as its secondary metric.
- DOTAv1 organization installs its 458 validation images directly under `images/` and retains both label layouts;
  its loader also accepts legacy `images/val` datasets. Local and archive organizers validate a
  complete staged root before atomic replacement, remove stale files on success, and preserve a
  recoverable backup if rollback fails.
- Model YAMLs use `file_cfg.filename` as the canonical MXQ artifact; the matching same-stem ONNX
  artifact is derived unless `onnx_filename` explicitly names an exception.
- Every model YAML declares `post_cfg.dataset`; postprocessing uses it with `task` to resolve the
  model's output taxonomy and class count.
- Use `obb` as the canonical vision task key for oriented bounding boxes; retain
  `oriented_bounding_boxes` as a compatibility alias and normalize it at CLI, benchmark, and
  evaluator boundaries.
- Preserve anchorless decoded-output layout provenance through NMS. If provenance is unavailable
  for an ambiguous tensor, normalize it as raw channels-first before candidates-first.
- Reuse `vision.utils.letterbox.LetterBoxGeometry` for shared forward and inverse letterbox
  geometry.
- Require `pre_cfg.LetterBox` for YOLO detection postprocessing, and use metadata-enabled semantic
  preprocessing that exposes original `img0_shape` and `ratio_pad`. Prediction restores spatial
  logits with this geometry before `argmax`, while validation loaders reuse it for targets.
- Normalize dense MXQ outputs before inverse letterboxing: upsample depth `[B, 1, H/4, W/4]` or
  `[B, H/4, W/4]` maps by four, accept baked-resize `[H, W, 1]` or `[B, H, W, 1]` maps without
  another resize, and convert Cityscapes `[H, W, 19]` or `[B, H, W, 19]` logits to NCHW before restoration
  and `argmax`. Preserve existing ONNX layouts and baked class maps; validate baked IDs as finite,
  integral, and in-range before converting to `int64`.
- Resolve dense compilation datasets from `post_cfg.dataset`: NYU Depth, ADE20K, or Cityscapes;
  sample calibration inputs from their organized `images/` directories.
- ADE20K organization atomically installs its 2,000 validation pairs with required
  `objectInfo150.txt` and `sceneCategories.txt` metadata after validating the complete staged root.
- Dense local organizers reject symlinked data and metadata and require resolved copy sources to
  remain within the resolved dataset root; readiness and organization reject dense managed roots
  reached through any symlinked path component.
- ADE20K semantic validation uses matched letterbox geometry, ignores source label `0`, and reports mIoU followed by
  pixel accuracy.
- Cityscapes organization selects only 500 validation PNG pairs from the two official train/validation/test ZIP
  packages; validation maps the 19 canonical source IDs to train IDs and ignores all other labels. Reject semantic
  validation taxonomies other than `ade20k` and `cityscapes`, including evaluator overrides that
  differ from `model.post_cfg.dataset`.
- Reuse ImageNet, COCO, DOTAv1, WiderFace, and dense validation or calibration roots only after
  checking taxonomy-specific layout, required metadata and targets, and the complete official split.
- Validate complete staged ImageNet, COCO, and WiderFace roots before atomic replacement, preserving
  an existing cache on failure; match WiderFace event/image identities exactly to `wider_face_val.mat`.
- `MobilintNPUBackend` hosts `N` `qbruntime.Model` slots; `max_batch_size` is the aggregate `N*K`
  capacity where `K` is the compiled MXQ batch axis. Slots are distributed round-robin across the
  unique devices referenced by the canonical target strings. A non-batch MXQ (`K==1`) with `B>1`
  fans out into `N=B` slots dispatched in parallel via `MobilintNPUBackend.infer_slot`.
- `dev_no` is syntactic sugar for the device-prefix component of the canonical target strings.
  Scalar pins one device; a list expands to multiple. Do not read `dev_no` at dispatch time —
  read `_target_cores_serialized` / `_target_clusters_serialized` (or the public
  `target_cores` / `target_clusters` accessors). The aggregate `target_cores` / `target_clusters`
  return the union across every covered device with device prefix dropped; use
  `target_cores_by_device` / `target_clusters_by_device` for per-device provenance.
- Backend target topology is accumulated on `NPUTargetSpecPending` at
  `MobilintNPUBackend._pending`. Each per-field setter records its raw override without
  normalizing and invalidates `MobilintNPUBackend._finalized`; the canonical `NPUTargetSpec` is
  materialized lazily on the next `_spec` read via `NPUTargetSpecPending.finalize`, which runs the
  single ordered pipeline (legacy migration → sibling drop → grain unification → off-mode drop →
  device-set consistency → `global8` coverage) once every accumulated override is visible.
  Setter order is irrelevant — the resolved spec depends on the *set* of overrides, not the
  sequence. `NPUTargetSpec.from_kwargs` remains the config-layer entry (JSON load) where eager
  normalization is unambiguous. Target-only override syncs `dev_no` to the target device set;
  `dev_no`-only override clears stale targets and re-expands sugar; both overridden → device-set
  consistency check surfaces mismatches on the next canonical read (not on the setter). Override
  intent flags are scoped to a single setter chain: every canonical read promotes
  `MobilintNPUBackend._pending` to a fresh `NPUTargetSpecPending` baseline (via
  `NPUTargetSpecPending.from_baseline`) so a subsequent setter chain — or any standalone runtime
  mutation — never inherits stale intent flags from the previous chain. Within a single chain
  (no mid-chain accessor read) accumulated overrides finalize as one atomic decision; across
  chains each chain sees a clean intent slate.
- Canonical NPU target wire form: `target_cores` items are `"d:c:k"` strings and
  `target_clusters` items are `"d:c"` strings. Legacy 2-part `c:k` cores, bare integer clusters,
  and `qbruntime.CoreId` / `Cluster` objects are silently migrated by
  `NPUTargetSpec.from_kwargs` (and its `_normalize_npu_target_kwargs` config-layer wrapper)
  using `dev_no` as the fallback prefix. Under `single`, `target_clusters` unfolds into every
  core of each cluster; under `multi` / `global4` / `global8`, `target_cores` folds up to
  `"d:c"` cluster prefixes and warns on partial coverage. `global8` requires both clusters on
  every covered device.
- `MobilintCache([m0, m1, ...], per_model_batch=K)` dualizes KV along `(model_idx, cache_id)`
  with total capacity `N*K` rows. Row `i` maps to `(i // K, i % K)`. Use `slot_of`, `model_of`,
  `group_by_model` for dispatch. `ensure_batch_size` beyond `N*K` is only allowed on the legacy
  single-Model hardware-batch path (`N==1`). `MobilintCache(model, batch_size=K)` is the shim
  for the historical `N=1, K=K` case.
- `MobilintBeamCache` enforces `N==1` — beam search bookkeeping tracks one active qbruntime
  cache. Multi-Model dispatch is a `MobilintCache`-only feature.
- Shared `MobilintModelMixin.decoder_forward` (BLIP text head) is `N==1` only — one blocking
  `mxq_model.infer` on slot 0 with no cross-slot routing / beam-cache reorder. `N>1` (e.g.
  `text_max_batch_size>K` on a `K==1` text MXQ) hard-fails with `NotImplementedError`; drop
  `--batch-size` or compile a `K>1` text MXQ. Guard is routed through
  `MultiSlotDispatcher.assert_single_slot` so every `N==1` caller shares one enforcement site.
- `MobilintNPUBackend.dispatcher` (`multi_slot_dispatch.MultiSlotDispatcher`) is the sole entry
  point for batched NPU dispatch. Owns `slot_of` routing, single-group fast path, multi-group
  `ThreadPoolExecutor` fan-out with worker-exception re-raise, NPU-time accounting (elapsed for
  single-group; wall time for multi-group so parallel work is not double-counted), and per-group
  merge. `modeling_utils._llm_forward_batch` collapses to a thin delegation.
- `MobilintNPUBackend.output_layout` (`"n_items"` / `"n_tokens"`) probed once from slot 0's
  `get_model_output_shape` (index `-2`: static -> `"n_items"`; `-1` -> `"n_tokens"` only when
  `k_per_model == 1`; `-1` + `K > 1` is ambiguous — the compiled batch axis can occupy
  position `-2` — so the probe returns `None` and defers to the runtime fallback). Consumed in
  `MultiSlotDispatcher._merge_group_outputs`; the old per-dispatch shape inference is gone.
  Ambiguous or missing shape probe falls back to inspecting an unambiguous runtime group and
  pins the answer via `_set_output_layout`; a wholly-ambiguous dispatch (every group is
  `n_rows == n_items == n_tokens`) hard-fails rather than silently defaulting to layout A.
  The dispatcher also cross-checks the cached layout against each dispatch's observed row
  count and re-pins the backend cache on disagreement, so a stale probe self-heals.
- On HBM `BadAlloc`, `MobilintNPUBackend.create` / `.launch` disposes every previously loaded
  slot and re-raises as `MobilintBackendAllocError` with `phase`, `slot`, `dev`,
  `succeeded_so_far`, `n_total`, `max_batch_size`, `k_per_model` context. Callers should lower
  `max_batch_size` or spread across more devices via `dev_no`.
- Keep TPS table and JSON output synchronized through `mblt_model_zoo/cli/tps_table.py`.
- Keep VLM non-batch tests under `image_text_to_text/non_batch`; matrix-runner Phase B owns the
  batch text-generation and image-text-to-text suites.
- Qwen3-VL video and per-prompt multi-image inputs require a dynamic-vision release
  (`MobilintQwen3VLConfig.dynamic_vision=True`); static-vision releases hard-fail those inputs
  from `MobilintQwen3VLProcessor.__call__` with `NotImplementedError` pointing at a
  dynamic-vision release. Batched single-image prompts are always allowed.
- The Qwen3-VL processor reads `config.dynamic_vision` in `from_pretrained` and mirrors it onto
  its video processor. Call `MobilintQwen3VLProcessor.sync_dynamic_vision_from_model(model)`
  only when a runtime `vision_mxq_path=` override diverges from the shipped config.
- EAGLE-3 speculative decoding (`mobilint/EAGLE3-Qwen3-4B` and siblings) loads through
  `AutoModelForCausalLM.from_pretrained(...)` as one release bundling base MXQ, draft MXQ, and
  FC stack. Qwen3 and Llama base families live under
  `mblt_model_zoo/hf_transformers/models/{qwen3_eagle3,llama_eagle3}/`
  (`MobilintQwen3Eagle3ForCausalLM`, `MobilintLlamaEagle3ForCausalLM`) and share
  `MobilintEagle3BaseModelMixin` / `MobilintEagle3DraftModelMixin`; embed_tokens + rotary_emb
  init lives in the mixins so every concrete `MobilintXxxEagle3ForCausalLM.__init__` is a thin
  wiring shim. Tune the draft budget with `GenerationConfig.num_assistant_tokens` (default `64`);
  Qwen3-4B measures best around `25`–`30`.
- `mblt_model_zoo/hf_transformers/utils/eagle3/tree_decoding.py::softmax_topk_cpu_torch` defaults
  to `auto`: slice by declared `TopKLogitsWarper` if present, else full-vocab softmax so
  `TopPLogitsWarper` computes its nucleus over the whole distribution. The slice-by-TopK path
  detects boundary ties (HF's strict-less-than `TopKLogitsWarper` keeps every logit equal to the
  k-th threshold; `torch.topk` drops tied entries at the boundary) via
  `(x >= threshold).sum(-1) > slice_size` and falls back to full-vocab when true, so it stays
  HF-equivalent even under ties. `full` forces full-vocab; `sliced` is a deprecated legacy
  top-``max_return_k`` renormalization that violates HF nucleus semantics for TopP-only and
  emits a warning. Toggle via `MBLT_EAGLE3_SOFTMAX_TOPK_MODE=auto|full|sliced` or
  `set_softmax_topk_mode(...)`. The `max_return_k` keyword (default `10`) is only a
  return-slice size, not a math slice. The greedy path never enters this function.
- Keep `prepare_logits_processor` in HF order (RepetitionPenalty → Temperature → TopK → TopP)
  so the auto slice-by-TopK path stays mathematically equivalent to full-vocab (modulo the
  boundary-tie fallback).
- EAGLE-3 root/next-token sampling goes through
  `tree_decoding.py::_sample_next_token_from_processor` (full-vocab softmax → `multinomial`).
  `softmax_topk_cpu_torch` is candidate-matching only; passing its top-N slice to `multinomial`
  renormalizes over 10 tokens and breaks temperature-only and `top_k > 10` configs.
  `evaluate_posterior` returns `sampled_indices=None` + raw next-root logits on the clean-accept
  branch (caller samples full-vocab) and top-N `(sample_p, sampled_indices)` on the
  rejection-adjusted branch (partial approximation retained for compatibility).
- Keep `evaluate_posterior` greedy argmax-first (`argmax(logits)[safe_positions]`) to avoid
  materializing the full `(n_cand, depth, vocab)` slice.
- EAGLE-3 speculative-decode rows in `tps measure` are `accept_steps`, `tokens_sum`,
  `tokens_per_step` (= `drafts_avg + 1`, matching `accept_length + 1` in the reference
  `eagle3MXQ.py`), and `draft_accept_ratio`; non-EAGLE-3 pipelines omit them.
- `tps measure` accepts `--print-output`, mutually exclusive `--enable-thinking` /
  `--disable-thinking` (Qwen3 chat template override), and `--temperature FLOAT` (`0` = greedy);
  chat templates apply to text prompts by default. `tps sweep` stays greedy. VLM (`--task
  image-text-to-text`) `tps measure` decode uses greedy `argmax` on the fake-prefill path; the
  CLI rejects `--temperature > 0` there with a clear error.
- On EAGLE-3, `_apply_eagle3_gen_kwargs` in `benchmark_utils.py` drops `min_new_tokens` and
  `pad_token_id` and sets `eos_token_id=None`, so `--decode N` becomes an upper bound and
  measured `num_decode` reflects actual generation.
- MXQ backends have known cross-process non-determinism; use
  `scripts/probe_mxq_determinism.py`, `scripts/probe_generate_same_process.py`, and
  `scripts/probe_warmup_stabilization.py` to reproduce. Warmup does not stabilize across
  processes; prefer same-process `--repeat N`.
- Read the nearest area README or `TEST.md` before modifying code or selecting validation.
- Preserve unrelated working-tree changes and report environment-dependent test limitations.
