# Claude Code Guide

@AGENTS.md

## Claude-Specific Notes

- Treat `AGENTS.md` as the canonical shared guidance.
- `.claude/skills/mblt-model-zoo/SKILL.md` is a symlink to its `.agents/skills/...` counterpart. Edit
  the `.agents` copy; there is no separate Claude version to keep in sync.
- Vision implementation guidance lives in `../mblt-vision-python`, and Transformers guidance
  (including the EAGLE-3 speculative-decoding workflow) lives in `../transformers-mblt`. Model Zoo
  retains only forwarding compatibility modules and CLI bridges; it must not ship copied Vision or
  Transformers code, YAMLs, tests, or benchmarks.
