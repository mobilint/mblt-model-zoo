---
trigger: always_on
description: Guidelines for formatting and structuring Markdown (.md) documents
---

# Markdown Style Guidelines

When creating or modifying Markdown (`.md`) files in this repository, follow these precise styling rules to ensure consistency and readability.

## 1. Headers
- Use ATX-style headers (e.g., `# Header 1`, `## Header 2`).
- Always leave a single blank line before and after a header, except for the document title (`# Header 1`) at the very top.
- Do not skip header levels (e.g., do not jump from `#` to `###`).

## 2. Lists
- Use hyphens (`-`) for unordered lists, not asterisks (`*`) or pluses (`+`).
- For nested lists, indent by exactly 2 spaces.
- Use periods after list items only if they are complete sentences.

## 3. Code Blocks
- Always specify a language identifier for fenced code blocks (e.g., ` ```python `).
- Use inline code formatting (backticks) for file names, variable names, functions, and terminal commands appearing within regular text.

## 4. Links and Images
- Prefer reference-style links for long documents, or inline links for short documents.
- Always provide descriptive `alt` text for images (e.g., `![System Architecture Diagram](./assets/arch.png)`).

## 5. Spacing and Line Length
- Ensure there is exactly one blank line between distinct elements (paragraphs, lists, code blocks).
- Avoid trailing whitespaces at the end of lines.
- No strict hard wrap for line lengths, but try to keep individual paragraphs concise.

## 6. Frontmatter
- If the document represents a rule or workflow for the agent, it must include YAML frontmatter at the top specifying its `description` and any target `paths` it applies to.