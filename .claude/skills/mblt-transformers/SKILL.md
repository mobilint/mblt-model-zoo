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
