# Claude Code Guide

@AGENTS.md

## Claude-Specific Notes

- Treat `AGENTS.md` as the canonical shared guidance.
- `.claude/skills/mblt-model-zoo/SKILL.md` and `.claude/skills/mblt-transformers/SKILL.md` are
  symlinks to their `.agents/skills/...` counterparts. Edit the `.agents` copy; there is no
  separate Claude version to keep in sync.
- Use `.claude/skills/mblt-transformers/SKILL.md` for the EAGLE-3 speculative-decoding workflow.
- Vision implementation guidance lives in `../mblt-vision-python`. Model Zoo retains only
  forwarding compatibility modules and CLI bridges; it must not ship copied Vision code or YAMLs.
