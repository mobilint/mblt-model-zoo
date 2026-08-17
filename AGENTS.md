---
description: Shared repository guidance for Codex, Claude Code, and other coding agents working on Mobilint Model Zoo.
paths:
  - "**"
---

# Mobilint Model Zoo Agent Guide

## Purpose and Precedence

This is the canonical repository guide. `CLAUDE.md` imports it so Codex and Claude Code use the
same rules. Shared workflow lives in `.agents/skills/mblt-model-zoo`; focused Vision and
Transformers skills live alongside it. Claude Code loads matching small entry points from
`.claude/skills/`. Follow more-specific `AGENTS.md` files in a subdirectory when present. User and
system instructions take precedence.

Keep this guide and `CLAUDE.md` synchronized for shared repository guidance so Codex and Claude
Code receive the same workflow requirements. The focused reusable skills are
`.agents/skills/mblt-vision` and `.agents/skills/mblt-transformers`; maintain matching Claude
entry points in `.claude/skills/`.

Before editing, run `git status --short`; preserve unrelated changes in a dirty worktree.

## Current Package Snapshot

- Package: `mblt-model-zoo`; read the current version from
  [`mblt_model_zoo.__version__`](mblt_model_zoo/__init__.py), the source of truth used by `pyproject.toml`.
  Supported Python: `>=3.10,<3.13`.
- Runtime dependencies are declared in `pyproject.toml`; install development tools with
  `pip install -e . --group dev`.
- Optional extras: `transformers`, `MeloTTS`, `onnxruntime`, `onnxruntime-gpu`, `qwen-asr`, and
  `qbcompiler`.
- Console scripts: `mblt-model-zoo` and `mblt-melotts-download`.
- The main CLI provides `predict` (aliases: `classify`, `detect`, `pose`, `segment`), `val`,
  `compile`, `tps`, `melo` (alias: `melotts`), and `melo-ui`. It delegates supported upstream
  Transformers commands when that extra is installed.

Treat `pyproject.toml`, CLI parsers, public package exports, and area README files as the source of
truth when this snapshot becomes stale.

## Repository Map

- `mblt_model_zoo/vision`: public vision API, model wrappers, dataset registry, evaluation, and
  task packages.
- `mblt_model_zoo/compile`: installable compilation API; vision compilation is optional.
- `mblt_model_zoo/hf_transformers`: Hugging Face integrations and benchmark utilities.
- `mblt_model_zoo/MeloTTS`: MeloTTS integration and text normalization.
- `mblt_model_zoo/cli`: installed CLI implementation.
- `tests`: pytest suites and shared NPU option helpers.
- `benchmark`: vision and Transformers benchmark scripts; `compile/` holds compatibility scripts
  and compilation documentation.

## General Engineering Rules

- Use four-space indentation, PEP 484 annotations, and Google-style docstrings for new or modified
  Python modules, classes, functions, and methods.
- Keep lines at 120 characters or fewer. Let Ruff manage imports and formatting.
- Group imports as standard library, third-party, then local.
- Catch specific exceptions and provide recovery-oriented error messages. Do not catch `Exception`
  unless immediately re-raising or deliberately adding context.
- Preserve local style in `mblt_model_zoo/hf_transformers` and `mblt_model_zoo/MeloTTS`: both are
  excluded from repository-wide Ruff checks.
- Write comments for non-obvious rationale, not mechanics. Format temporary notes as
  `TODO(username): description`.

## Area-Specific Contracts

### Vision

- Prefer `MBLT_Engine` and task-subpackage imports in new code. Legacy top-level model imports
  remain supported and must stay synchronized with task-package `__all__` exports so
  `vision.list_models()` continues to work.
- Preserve public compatibility arguments such as `local_path`, `model_type`, `infer_mode`, and
  `product`. The YAML registry ignores `product`; select a non-default artifact with `model_cls`,
  `model_type`, or `model_path` instead.
- Prefer `model_path` in new APIs, tests, and docs. `mxq_path` and `onnx_path` are compatibility
  aliases. Framework inference recognizes local `.mxq` and `.onnx` suffixes; retain the fail-fast
  error for an explicit framework that conflicts with a local suffix.
- In model YAML `file_cfg`, use `filename` as the canonical MXQ artifact and let the loader derive
  the same-stem ONNX filename. Set `onnx_filename` only for a Hub artifact that has a different name.
- Every model YAML `post_cfg` must declare `dataset` as the output taxonomy and validation-dataset
  identifier. Dataset-aware postprocessing resolves class counts from the `(dataset, task)` pair.
- The supported discovery tasks are `image_classification`, `depth_estimation`, `object_detection`,
  `instance_segmentation`, `semantic_segmentation`, `obb`, `pose_estimation`, and `face_detection`.
  Use `obb` as the canonical oriented-bounding-box task key while retaining
  `oriented_bounding_boxes` as a compatibility alias.
- Keep model configuration shape (`model_cfg`, `pre_cfg`, and `post_cfg`) stable unless changing
  the public contract deliberately.
- Preserve anchorless decoded-output layout provenance through NMS. When provenance is unavailable
  and a tensor shape is ambiguous, normalize it as raw channels-first before candidates-first.
- Keep shared forward and inverse letterbox geometry in
  `mblt_model_zoo/vision/utils/letterbox.py`; task-specific code selects interpolation and padding
  values without duplicating resize, border, or crop calculations.
- YOLO detection postprocessors require `pre_cfg.LetterBox`. Semantic validation preprocessors
  must return image data with original `img0_shape` and LetterBox `ratio_pad` metadata. Semantic
  prediction passes both values through postprocessing so spatial logits are inverse-letterboxed
  before `argmax`; loaders use the same metadata to apply identical target geometry.
- Normalize dense MXQ outputs before inverse letterboxing: bilinearly upsample depth
  `[B, 1, H/4, W/4]` or `[B, H/4, W/4]` maps by four, accept baked-resize `[H, W, 1]` or
  `[B, H, W, 1]` depth maps without another resize, and convert Cityscapes `[H, W, 19]` or `[B, H, W, 19]` logits to
  NCHW before bilinear restoration and `argmax`. Preserve full-resolution ONNX depth, NCHW
  semantic logits, and baked class maps. Accept floating baked maps only when every value is
  finite, integral, and within the configured class range; validate before converting to `int64`.

### Vision Datasets and Compilation

- Keep validation datasets in `mblt_model_zoo/vision/datasets/*.yaml`. Use `path`, `val`, optional
  `names`, and the repository `tasks` and `download` metadata; resolve defaults with
  `get_dataset_config_for_task()` rather than duplicating URLs or paths.
- Keep the NYU Depth organizer validation-only: install its 654 paired samples as `images/` and `depth/` directly
  under its output root.
- Keep the ADE20K organizer validation-only: stage and validate its 2,000 paired samples as flat
  `images/` and `annotations/` directories with required `objectInfo150.txt` and
  `sceneCategories.txt` metadata, then atomically replace the managed root.
- Keep the Cityscapes organizer ZIP-only and validation-only: select exactly 500 paired images and `gtFine_labelIds`
  masks from the official `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` packages, then install lossless
  flat `images/` and `annotations/` PNG pairs.
- Reject symlinked dense dataset files and metadata from local extracted sources, and require every
  resolved copy source to remain within the resolved dataset root. Do not reuse dense managed roots
  containing symlinked data, metadata, or layout directories, or roots reached through a symlinked
  ancestor. Organizers must reject those output paths before writing.
- Reuse an organized validation or calibration root only when its taxonomy-specific layout,
  required metadata and targets, and full official validation sample count are all valid. Apply
  this consistently to ImageNet, COCO, DOTAv1, WiderFace, and dense datasets.
- Validate complete staged ImageNet, COCO, and WiderFace roots before atomically replacing managed
  output so failed readiness repair preserves a valid cache and successful repair removes stale
  files. Match WiderFace event and image identities exactly to `wider_face_val.mat`, not only its
  aggregate counts.
- Depth-estimation validation stretches RGB and depth targets to the configured input size, median-aligns each
  prediction, pools statistics over target-valid NYU Depth V2 pixels, rejects non-finite predictions at those
  pixels, and reports `delta1` as the primary score with `abs_rel` and `rmse` as auxiliary metrics.
- ADE20K semantic-segmentation validation applies matching letterbox geometry to images and masks, ignores source
  label `0`, maps labels `1..150` to model classes `0..149`, and reports mIoU as primary with pixel accuracy secondary.
- Cityscapes semantic-segmentation validation maps source IDs `7,8,11,12,13,17,19..28,31..33` to classes `0..18`,
  ignores other IDs, and reports mIoU as primary with pixel accuracy secondary. Reject semantic
  validation taxonomies other than `ade20k` and `cityscapes`, and reject evaluator taxonomy
  overrides that differ from `model.post_cfg.dataset`.
- Keep `eval_imagenet()` numerically compatible by returning Top-1 as a float. Use
  `eval_imagenet_metrics()` for structured Top-1 primary and Top-5 secondary metrics.
- Keep evaluator result objects aligned on `primary_score` and `secondary_score`. Preserve numeric
  `eval_coco()` compatibility while using `eval_coco_metrics()` for structured mAP50-95 and mAP50.
- DOTAv1 validation reports rotated mAP50-95 as the primary metric and rotated mAP50 as the secondary metric.
- The standard Vision benchmark runner supports depth and semantic segmentation. Record `delta1`,
  `abs_rel`, and `rmse` for depth; dispatch semantic evaluation from `post_cfg.dataset` and record
  `miou` and `pixel_accuracy`.
- The Vision benchmark runner auto-detects `.onnx` from `--model-path` before expanding core modes;
  record one neutral `onnx` target even when `--core-mode all` is requested.
- Preserve evaluator layouts. DOTAv1 stores its validation images directly in `images/` and may
  use `labels/val_original`, which retains difficult-object filtering. Its loader also accepts
  legacy validation images under `images/val`. Stage and validate complete local and archive
  DOTAv1 roots before atomically replacing the cache; successful replacement removes stale files,
  failed validation preserves the cache, and failed rollback preserves a recoverable backup.
- Normalize Vision task aliases at evaluator boundaries. CLI and benchmark DOTAv1 evaluation must
  accept `oriented_bounding_boxes` from external model configuration as the compatibility spelling
  of canonical `obb`.
- Expose a seed with default `0` for vision APIs, CLIs, benchmarks, and compatibility helpers that
  sample or otherwise use randomness.
- Keep qbcompiler imports inside the compilation path. Base imports, vision imports, the compile
  module import, and non-compile CLI commands must work without `qbcompiler` installed.
- Compilation accepts exactly one entry level: `data_path` (organize, sample, preprocess),
  `subset_path` (skip organization and sampling), or `calib_data_path` (validated `.npy` tensors;
  skip all preparation). Keep default models under `~/.mblt_model_zoo` and datasets under
  `~/.mblt_model_zoo/datasets`.
- Compilation resolves dense calibration datasets from `post_cfg.dataset`: NYU Depth for depth
  estimation and ADE20K or Cityscapes for semantic segmentation. Sample RGB images from the
  organized `images/` directory; targets are validation metadata, not compiler inputs.

### Transformers and MeloTTS

- Install the matching optional extra before running integration tests.
- `MobilintNPUBackend` hosts `N` `qbruntime.Model` slots. `max_batch_size` is the aggregate batch
  capacity `N * K`, where `K` is the compiled MXQ batch axis probed from slot 0; the backend
  launches `N = ceil(max_batch_size / K)` slots and distributes them round-robin across the unique
  devices referenced by the canonical target strings. For a non-batch MXQ (`K == 1`) with `B > 1`,
  sw-batch fans a logical batch across `N = B` slots that dispatch in parallel via
  `MobilintNPUBackend.infer_slot`; for a batched MXQ (`K > 1`) hardware batching is reused until
  `N * K >= max_batch_size`. Beam search paths stay `N = 1`.
- `dev_no` is syntactic sugar for the device-prefix component of the canonical target strings. A
  scalar pins one device; a list expands to multiple devices. Do not read `dev_no` at dispatch
  time — use the canonical `_target_cores_serialized` / `_target_clusters_serialized` lists (or
  the public `target_cores` / `target_clusters` accessors) so multi-device backends behave
  correctly. The aggregate `target_cores` / `target_clusters` accessors return the union across
  every covered device but drop the device prefix from the return type; callers that need
  per-device provenance should read the sibling `target_cores_by_device` /
  `target_clusters_by_device` mappings or the canonical serialized lists.
- Backend target topology is accumulated on `NPUTargetSpecPending` at `MobilintNPUBackend._pending`.
  Each per-field setter (`dev_no`/`core_mode`/`target_cores`/`target_clusters`) records its raw
  override on the pending log without normalizing and invalidates `MobilintNPUBackend._finalized`.
  The canonical `NPUTargetSpec` is materialized lazily on the next read of the `_spec` property
  via `NPUTargetSpecPending.finalize`, which runs the single ordered pipeline (legacy migration →
  sibling drop → grain unification → off-mode drop → device-set consistency → `global8` coverage)
  once every accumulated override is visible. Setter order is therefore irrelevant — the resolved
  canonical spec depends only on the *set* of accumulated overrides, not the sequence HF used to
  apply them. `NPUTargetSpec.from_kwargs` remains the config-layer entry point (JSON load) where
  eager normalization is unambiguous because the whole payload arrives at once. When a caller
  overrides only targets, `dev_no` syncs to the target device set at finalize; when a caller
  overrides only `dev_no`, stale targets are cleared and re-expanded from the new device sugar;
  when both are overridden, the device-set consistency check catches genuine mismatches on the
  next canonical read (not on the setter itself).
- The canonical NPU target wire form is fully-qualified: `target_cores` entries are `"d:c:k"`
  strings and `target_clusters` entries are `"d:c"` strings. Legacy 2-part `c:k` cores, bare
  integer clusters, and `qbruntime.CoreId` / `Cluster` objects are silently migrated to the
  canonical form by `NPUTargetSpec.from_kwargs` (and its `_normalize_npu_target_kwargs` config-
  layer wrapper) using `dev_no` as the fallback prefix; no explicit migration step is required
  for stored configs. Under `single` core mode, the config layer expands `target_clusters` into
  every core of each cluster; under `multi` / `global4` / `global8`, it folds `target_cores` up
  to their unique `"d:c"` cluster prefixes and warns when a partial cluster is rounded up.
  `global8` requires both clusters `0` and `1` on every device covered by the target set.
- `MobilintCache([m0, m1, ...], per_model_batch=K)` dualizes KV state along
  `(model_idx, cache_id)` with `N = len(mxq_models)` and total capacity `N * K` rows. Row `i`
  maps to `(i // K, i % K)`; `slot_of`, `model_of`, and `group_by_model` expose the routing so
  upstream dispatch groups rows by owning Model and issues one `Model.infer` per Model.
  `ensure_batch_size` beyond `N * K` is only supported on the legacy single-Model
  hardware-batch path (`N == 1`). The legacy `MobilintCache(model, batch_size=K)` constructor
  still works as a shim for the historical `N = 1, K = K` case; do not pass both
  `per_model_batch` and `batch_size` in the same call.
- `MobilintBeamCache` (Whisper and other encoder-decoder beam searches) enforces `N == 1` because
  the beam bookkeeping tracks one active qbruntime cache; constructing it with more than one
  Model raises `NotImplementedError`. Use `MobilintCache` for multi-Model dispatch.
- The shared `MobilintModelMixin.decoder_forward` (BLIP text head and any future encoder-decoder
  backend that inherits it) is `N == 1` only. It issues one blocking `mxq_model.infer` on slot 0
  and has no cross-slot routing or beam-cache reorder, so growing the backend to `N > 1` via
  `text_max_batch_size > K` on a `K == 1` text MXQ is rejected with a clear
  `NotImplementedError`. Users needing higher batched throughput should either drop the CLI
  `--batch-size` request or compile a batched (`K > 1`) text MXQ so slot-0 hardware batching
  services the load. The guard is routed through `MultiSlotDispatcher.assert_single_slot` so
  every caller that requires `N == 1` shares one enforcement site.
- Batched multi-slot NPU dispatch is centralized in
  `mblt_model_zoo/hf_transformers/utils/multi_slot_dispatch.py::MultiSlotDispatcher`. It owns
  slot routing (`slot_of`, `k_per_model`), single-vs-multi-group dispatch (single-group fast path;
  multi-group `ThreadPoolExecutor` with `max_workers = len(groups)` and worker-exception
  re-raise), NPU-time accounting (elapsed for single-group, wall time for multi-group so parallel
  work is not double-counted), and merging per-group outputs back into caller row order.
  `MobilintNPUBackend.dispatcher` is the sole entry point; `modeling_utils._llm_forward_batch`
  and every downstream (Qwen3-VL deepstack included) delegate here rather than reimplementing
  the closure.
- `MobilintNPUBackend.output_layout` (`"n_items"` or `"n_tokens"`) is a fixed property of the
  compiled MXQ probed once from slot 0's `get_model_output_shape` — a static value at index
  `-2` marks the artifact as per-item last row. A `-1` at index `-2` maps to per-token flat
  (`"n_tokens"`) only when `k_per_model == 1`; on a `K > 1` batched MXQ the compiled batch
  axis can also be reported dynamic at that position, so the probe cannot disambiguate and
  returns `None` (defers to the runtime fallback). The layout drives
  `MultiSlotDispatcher._merge_group_outputs`; the old per-dispatch shape inference is gone.
  When the compile-time probe is ambiguous or unavailable the dispatcher inspects an
  unambiguous runtime group and pins the answer via `_set_output_layout` for the remainder
  of the process, and hard-fails if every group in the dispatch collapses to
  `n_rows == n_items == n_tokens`. Belt-and-suspenders: the dispatcher also cross-checks a
  cached layout against every dispatch's observed row count and re-pins the backend cache
  when they disagree, so a stale/incorrect compile-time probe self-heals on the first
  unambiguous dispatch.
- On HBM `BadAlloc` during `create` or `launch`, `MobilintNPUBackend` disposes every previously
  loaded slot and re-raises the underlying `qbruntime.QbRuntimeError` as
  `MobilintBackendAllocError` with `phase`, `slot`, `dev`, `succeeded_so_far`, `n_total`,
  `max_batch_size`, and `k_per_model` context. Callers should lower `max_batch_size` or spread
  the workload across additional devices via `dev_no` (or explicit fully-qualified target
  strings) rather than retrying on the same target set.
- Keep `mblt-model-zoo tps` table labels, JSON keys, units, and extraction behavior centralized in
  `mblt_model_zoo/cli/tps_table.py`; update its schema and focused CLI TPS tests together.
- Keep non-batch VLM tests under `tests/transformers/image_text_to_text/non_batch`. Route batch
  text-generation and image-text-to-text suites through the serial Phase B in
  `scripts/test_transformers_matrix.py`, where they can use the required NPU core allocation.
- Qwen3-VL treats the vision MXQ, text MXQ, and processor as one release. `MobilintQwen3VLConfig`
  declares the top-level `dynamic_vision` bool; the shipped `config.json` selects which release a
  caller loaded. A dynamic-vision release (`dynamic_vision=True`) accepts video and per-prompt
  multi-image inputs. A static-vision release (`dynamic_vision=False`) supports at most one image
  per prompt and rejects video and per-prompt multi-image inputs. Batched single-image prompts
  are always allowed. `MobilintQwen3VLProcessor.from_pretrained` reads `config.dynamic_vision`
  and stays in lock-step with its video processor.
- Qwen3-VL static-vision releases hard-fail from `MobilintQwen3VLProcessor.__call__` with
  `NotImplementedError` when the caller passes a video input or more than one image per prompt.
  The message points the caller at a dynamic-vision release; keep the guard and message in
  `processing_qwen3_vl.py` as the contract statement.
- Call `MobilintQwen3VLProcessor.sync_dynamic_vision_from_model(model)` only when a runtime
  loader override (for example `vision_mxq_path=`) points at an MXQ whose signature
  (static/dynamic) does not match the shipped `config.dynamic_vision`. The helper adopts the
  vision submodule's detected signature (`visual._uses_dynamic_vision`) and re-syncs the video
  processor.
- Video decoding in the Qwen3-VL release requires the `transformers` extra's `torchcodec`
  dependency; validate video paths only against a dynamic-vision release.
- EAGLE-3 speculative decoding is a supported release family loaded through
  `AutoModelForCausalLM.from_pretrained(...)` (for example `mobilint/EAGLE3-Qwen3-4B`), which
  binds the base MXQ, one-block draft MXQ, and FC stack as one release. The draft-tree budget is
  `GenerationConfig.num_assistant_tokens` (defaults to `64` in
  `mblt_model_zoo/hf_transformers/utils/generation_utils.py`); the Qwen3-4B release measures best
  in the `25`–`30` range, where the Hugging Face default of `49` costs more iteration latency than
  its extra acceptance recovers.
- `mblt_model_zoo/hf_transformers/utils/eagle3/tree_decoding.py::softmax_topk_cpu_torch` runs in
  one of three modes. Default `auto` dispatches per call: when the processor list contains a
  `TopKLogitsWarper`, slice the raw logits to the declared top-K first and apply the processor
  list on that slice (HF `_get_logits_warper` order Temperature → TopK → TopP makes the slice
  mathematically identical to the full-vocab path while skipping the full-vocab `exp`), *except*
  when boundary ties push part of the active support outside the slice — HF's `TopKLogitsWarper`
  uses a strict-less-than filter and keeps every entry equal to the k-th threshold, while
  `torch.topk` drops arbitrary tied entries at the boundary, so the slice path detects this case
  cheaply (`(x >= threshold).sum(-1) > slice_size`) and falls back to the full-vocab helper to
  preserve HF equivalence; otherwise take the full-vocab path so a bare `TopPLogitsWarper` still
  determines its nucleus from the whole distribution. `full` forces the full-vocab path as a manual override.
  `sliced` is a deprecated back-compat mode that always renormalizes over a top-``max_return_k``
  slice and emits a warning; keep it only for A/B reproducibility. Toggle at import through
  `MBLT_EAGLE3_SOFTMAX_TOPK_MODE=full|sliced|auto` or programmatically through
  `set_softmax_topk_mode(...)`. The `max_return_k` argument (default `10`) is a return-slice
  size for downstream candidate matching, not the math slice; do not treat it as an implicit
  TopK. The greedy path never invokes this function because `prepare_logits_processor` returns
  `None` for `temperature<=1e-5`.
- Keep `prepare_logits_processor` in HF order (RepetitionPenalty → Temperature → TopK → TopP)
  so that `softmax_topk_cpu_torch` can safely apply the list on top of a TopK slice.
- Route EAGLE-3 root/next-token sampling through
  `mblt_model_zoo/hf_transformers/utils/eagle3/tree_decoding.py::_sample_next_token_from_processor`
  so it draws from the full-vocab HF-processed distribution (softmax over the whole vocab, then
  `torch.multinomial`). `softmax_topk_cpu_torch` is candidate-matching only — feeding its top-N
  slice to `torch.multinomial` renormalizes over 10 tokens and zeroes every other token, which
  breaks temperature-only and `top_k > 10` configurations. The `evaluate_posterior` sampling
  return schema now distinguishes clean-accept (raw next-root logits with `sampled_indices=None`)
  from rejection-adjusted (top-N renormalized `sample_p` + aligned `sampled_indices`); the
  rejection-adjusted branch remains a partial approximation of true rejection sampling until a
  full-vocab algorithm lands, and only the clean-accept branch samples the full processed
  distribution.
- Keep the argmax-first shape in `evaluate_posterior` greedy: compute `argmax(logits)` and index
  with `safe_positions` rather than fancy-indexing the `(n_cand, depth, vocab)` logits tensor,
  which materializes the full slice.
- EAGLE-3 speculative-decode rows in `mblt-model-zoo tps measure` are `accept_steps`,
  `tokens_sum`, `tokens_per_step` (= `drafts_avg + 1`, matching `accept_length + 1` in the
  reference `eagle3MXQ.py`), and `draft_accept_ratio`. Non-EAGLE-3 pipelines omit these rows
  automatically; keep the schema centralized in `mblt_model_zoo/cli/tps_table.py`.
- `mblt-model-zoo tps measure` exposes `--print-output` for diagnostic decoded text, mutually
  exclusive `--enable-thinking`/`--disable-thinking` to override the Qwen3 chat template
  `enable_thinking` flag, and `--temperature FLOAT` (`0.0` keeps greedy; any `>0` enables
  `do_sample=True`). Chat templates apply to text prompts by default. `tps sweep` stays greedy so
  its numbers remain comparable. VLM (`--task image-text-to-text`) `tps measure` decode is
  measured with a greedy `torch.argmax` on the fake-prefill decode path; the CLI rejects
  `--temperature > 0` there with a clear error rather than silently ignoring it.
- On EAGLE-3 pipelines the TPS measurement path in
  `mblt_model_zoo/hf_transformers/utils/benchmark_utils.py::_apply_eagle3_gen_kwargs` strips
  `min_new_tokens` and `pad_token_id` and sets `eos_token_id=None` so `generate` honors the real
  EOS in `config.json`; the measured `num_decode` reflects the tokens actually produced, and
  `--decode N` becomes an upper bound. Non-speculative pipelines keep exact-`N` semantics.
- MXQ backends exhibit known cross-process non-determinism. Use `scripts/probe_mxq_determinism.py`,
  `scripts/probe_generate_same_process.py`, and `scripts/probe_warmup_stabilization.py` to
  reproduce and isolate; warmup does not stabilize outputs across processes (verified by the
  warmup probe). For stable benchmarks, prefer same-process `--repeat N`.
- Start with the narrowest test file or documented `-k` selection. Use
  `pytest tests/transformers --full-matrix` only for a release or pre-merge matrix; use `-x` while
  iterating.
- Reuse shared NPU options and `tests.npu_backend_options.build_vision_engine_kwargs()` rather than
  adding divergent hardware flags or engine keyword bundles.
- Hardware, downloaded models, and external data may be unavailable. Run static or focused checks
  that are safe locally and state any limitation.

## Documentation and Validation

- Keep public documentation aligned with `MBLT_Engine`, `list_models()`, the shared
  `--model-path` option, and framework auto-detection.
- When CLI behavior, commands, aliases, or options change, update the README CLI guide and parser
  help together. Verify `mblt-model-zoo -h` and each affected command's `-h` output reflect the
  current interface.
- When a package update changes a durable fact—such as version support, dependencies or extras,
  public APIs, CLI commands, repository layout, validation, or workflow—update this guide and the
  applicable Codex and Claude skills in the same change. Reflect the change concisely in
  `CLAUDE.md` as well.
- Use ATX headings, one blank line between blocks, hyphen lists, language-tagged code fences, and
  concise paragraphs. Add `description` and `paths` YAML frontmatter to reusable agent rules or
  workflows.
- Follow the nearest guide before selecting a test:
  `tests/vision/TEST.md`, `tests/transformers/TEST.md`, `tests/MeloTTS/TEST.md`,
  `benchmark/vision/README.md`, or `benchmark/transformers/README.md`.
- Run the smallest meaningful validation. For a documentation-only change, check links, headings,
  and `git diff --check`; do not run hardware-bound suites unnecessarily.
- For Python changes, use the targeted relevant pytest file, then run
  `pre-commit run --files <touched files>` when available. Never bypass hooks with `--no-verify`.

## Git Safety

- Do not revert, format, or regenerate unrelated files.
- Keep commits focused and use Conventional Commit subjects under 50 characters in the imperative
  mood.
- Do not add generated artifacts, model weights, caches, or benchmark output unless the task
  explicitly requires them.
