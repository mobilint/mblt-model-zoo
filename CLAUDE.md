# Claude Code Guide

@AGENTS.md

## Claude-Specific Notes

- Treat `AGENTS.md` as the canonical shared guidance.
- The shared Claude skill entry point is `.claude/skills/mblt-model-zoo/SKILL.md`; its content is
  maintained in `.agents/skills/mblt-model-zoo/SKILL.md`.
- Use `.claude/skills/mblt-transformers/SKILL.md` for the EAGLE-3 speculative-decoding workflow.
- Vision implementation guidance lives in `../mblt-vision-python`. Model Zoo retains only
  forwarding compatibility modules and CLI bridges; it must not ship copied Vision code or YAMLs.
