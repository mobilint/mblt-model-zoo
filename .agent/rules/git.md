---
trigger: always_on
description: Git conventions and rules for the repository
---

# Git Rules

## Pre-commit Hook (CRITICAL)

- **Never skip pre-commit hooks** (`--no-verify` is forbidden).
- After `.pre-commit-config.yaml` exists, run `pre-commit install` to register the git hook.
- If pre-commit fails, fix the underlying issue and create a new commit.

## Commit Message Guidelines

- **Use Conventional Commits**: Prefix the commit message with a type (e.g., `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`).
- **Imperative Mood**: Write the subject line in the imperative mood (e.g., `feat: add user authentication` instead of `added user authentication`).
- **Subject Length**: Keep the summary line concise (under 50 characters).
- **Detailed Body**: Use the body of the commit to explain the what and why if further explanation is needed, separated from the title by a single blank line.

## General Practices

- **Atomic Commits**: Keep commits small, logical, and focused on a single change.