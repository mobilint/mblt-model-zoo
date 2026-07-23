# Claude Code Guide

@AGENTS.md

## Claude-Specific Notes

- Treat `AGENTS.md` as the canonical repository guidance. When a durable workflow changes, update
  it alongside this guide and both Codex and Claude skills.
- Keep this guide synchronized with `AGENTS.md` for shared repository guidance so Claude Code and
  Codex receive the same workflow requirements.
- The Claude Code skill entry point is `.claude/skills/mblt-model-zoo/SKILL.md`; its shared content
  is maintained in `.agents/skills/mblt-model-zoo/SKILL.md`.
- NYU Depth organization installs only its 654 validation image/depth pairs as `images/` and `depth/` at the output root.
- DOTAv1 organization installs its 458 validation images directly under `images/` and retains both label layouts.
- Read the nearest area README or `TEST.md` before modifying code or selecting validation.
- Preserve unrelated working-tree changes and report environment-dependent test limitations.
