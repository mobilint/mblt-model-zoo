---
trigger: always_on
---

# Code Style Guide

**Paths**: `mblt_model_zoo/vision`, `tests/vision`, `benchmark/vision`

## 1. Indentation & Formatting

- **Indentation**: Use **Tabs** for indentation.
- **Line Length**: Max 120 characters.
- **'Sadface' Style**: For multi-line function or class definitions, the closing parenthesis and colon MUST be on a separate line, indented with one extra level of indent.

**Example ('Sadface' & Tabs):**

```python
def very_long_function_name(
		arg_one: str,
		arg_two: int,
) :
	# Function body indented with tabs
	pass
```

## 2. Typing & Docstrings

- **Typing**: Use PEP484 type annotations for all function/method signatures.
- **Docstrings**: Use **Google Style**. Do not duplicate type information or default values in the docstring if they are already in the signature.

## 4. Error Handling

- Use `try/except` blocks with specific exceptions. Avoid catching `Exception` unless absolutely necessary.
- Raise informative errors with clear messages.

## 5. Imports & Tools

- **Imports**: Group imports (stdlib, third-party, local) and follow `isort` rules.
- **Formatter**: Use **Black** (configured for 120 chars and compatible with these rules).