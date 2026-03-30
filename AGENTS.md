# Agent Guidelines

## Code Style Guide

**Paths**: `mblt_model_zoo/vision`, `tests/vision`, `benchmark/vision`

### 1. Indentation & Formatting

- **Indentation**: Use **4 Spaces** for indentation.
- **Line Length**: Max 120 characters.
- **Multi-line Formatting**: Defer to Black's default styling for multi-line function or class definitions (closing parenthesis matches the indentation of the opening line).

**Example (Ruff/Black Style):**

```python
def very_long_function_name(
    arg_one: str,
    arg_two: int,
):
    # Function body indented with 4 spaces
    pass
```

### 2. Typing & Docstrings

- **Typing**: Use PEP484 type annotations for all function/method signatures.
- **Docstrings**: Use **Google Style**. Do not duplicate type information or default values in the docstring if they are already in the signature.

### 3. Error Handling

- Use `try/except` blocks with specific exceptions. Avoid catching `Exception` unless absolutely necessary.
- Raise informative errors with clear messages.

### 4. Imports & Tools

- **Imports**: Group imports (stdlib, third-party, local) and follow `ruff` rules (isort compatible).
- **Formatter**: Use **Ruff** (configured for 120 chars and compatible with these rules).

---

## Comment Style Guide


**Paths**: `**/*.py`

### 1. Docstrings

Every module, class, and function must have a docstring. Use **Google Style** docstrings.

- **Modules**: Describe the purpose of the module at the top of the file.
- **Classes**: Describe the class and its attributes.
- **Functions/Methods**: Describe the purpose, arguments (Args), return value (Returns), and any exceptions raised (Raises).

**Good Example:**

```python
def normalize_image(image: np.ndarray, mean: list, std: list) -> np.ndarray:
    """Normalizes an image using mean and standard deviation.

    Args:
        image: The input image as a numpy array.
        mean: A list of mean values for each channel.
        std: A list of standard deviation values for each channel.

    Returns:
        The normalized image as a numpy array.
    """
    return (image - mean) / std
```

### 2. Type Hinting

Always use type hints for function arguments and return values. This improves readability and allows for better static analysis.

### 3. Inline Comments

- Use inline comments sparingly.
- Explain **why** something is done, not **how**. The code should be clear enough to explain the "how".
- Avoid obvious comments like `x = x + 1  # Increment x`.

### 4. TODOs

Use `TODO(username): description` for temporary notes or planned improvements.

---

## Git Rules

### Pre-commit Hook (CRITICAL)

- **Never skip pre-commit hooks** (`--no-verify` is forbidden).
- After `.pre-commit-config.yaml` exists, run `pre-commit install` to register the git hook.
- If pre-commit fails, fix the underlying issue and create a new commit.

### Commit Message Guidelines

- **Use Conventional Commits**: Prefix the commit message with a type (e.g., `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`).
- **Imperative Mood**: Write the subject line in the imperative mood (e.g., `feat: add user authentication` instead of `added user authentication`).
- **Subject Length**: Keep the summary line concise (under 50 characters).
- **Detailed Body**: Use the body of the commit to explain the what and why if further explanation is needed, separated from the title by a single blank line.

### General Practices

- **Atomic Commits**: Keep commits small, logical, and focused on a single change.

---

## Markdown Style Guidelines

When creating or modifying Markdown (`.md`) files in this repository, follow these precise styling rules to ensure consistency and readability.

### 1. Headers

- Use ATX-style headers (e.g., `# Header 1`, `## Header 2`).
- Always leave a single blank line before and after a header, except for the document title (`# Header 1`) at the very top.
- Do not skip header levels (e.g., do not jump from `#` to `###`).

### 2. Lists

- Use hyphens (`-`) for unordered lists, not asterisks (`*`) or pluses (`+`).
- For nested lists, indent by exactly 2 spaces.
- Use periods after list items only if they are complete sentences.

### 3. Code Blocks

- Always specify a language identifier for fenced code blocks (e.g., ` ```python `).
- Use inline code formatting (backticks) for file names, variable names, functions, and terminal commands appearing within regular text.

### 4. Links and Images

- Prefer reference-style links for long documents, or inline links for short documents.
- Always provide descriptive `alt` text for images (e.g., `![System Architecture Diagram](./assets/arch.png)`).

### 5. Spacing and Line Length

- Ensure there is exactly one blank line between distinct elements (paragraphs, lists, code blocks).
- Avoid trailing whitespaces at the end of lines.
- No strict hard wrap for line lengths, but try to keep individual paragraphs concise.

### 6. Frontmatter

- If the document represents a rule or workflow for the agent, it must include YAML frontmatter at the top specifying its `description` and any target `paths` it applies to.
