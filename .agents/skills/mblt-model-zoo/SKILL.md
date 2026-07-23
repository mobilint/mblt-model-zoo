---
name: mblt-model-zoo
description: >-
  Follow shared Mobilint Model Zoo workflow for repository-wide, CLI, MeloTTS, and documentation
  changes. Use the mblt-vision or mblt-transformers skill for area-specific work.
---

# Mobilint Model Zoo

## Start Here

1. Read `AGENTS.md`; it is the canonical shared agent guide and `CLAUDE.md` imports it.
2. Run `git status --short` before changing files.
3. Read `pyproject.toml` and the nearest area guide before choosing dependencies or validation:
   - MeloTTS: `tests/MeloTTS/TEST.md` and `mblt_model_zoo/MeloTTS/README.md`.
4. For vision models, datasets, evaluation, or compilation, also use `mblt-vision`.
5. For `mblt_model_zoo/hf_transformers` or Transformers benchmarks, also use
   `mblt-transformers`.

## Shared Surface

- Use the installed `mblt-model-zoo` CLI for package behavior. Its native commands are `predict`,
  `val`, `compile`, `tps`, `melo`, and `melo-ui`; `classify`, `detect`, `pose`, `segment`, and
  `melotts` are aliases.

## Preserve Shared Contracts

- When a package update changes a durable public fact or workflow, update `AGENTS.md`, this skill,
  the relevant area skill, `CLAUDE.md`, and the matching Claude skill entry point in the same change.
  Keep shared guidance concise.
- Do not force formatting standards on `hf_transformers` or `MeloTTS`; follow local style.

## Validate Proportionately

- Run one targeted test file or `-k` selection first; append `-x` while fixing failures.
- Expect NPU tests and model-download tests to be environment-dependent. Report unavailable
  hardware, artifacts, or extras rather than broadening the test run.
- Run `pre-commit run --files <touched files>` for Python edits when hooks are available. For docs,
  run `git diff --check` and inspect rendered Markdown structure.

## Avoid

- Do not revert unrelated user work, commit generated outputs, or bypass pre-commit hooks.
