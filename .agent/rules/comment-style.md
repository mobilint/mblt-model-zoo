---
trigger: always_on
---

# Comment Style Guide

paths: "**/*.py"

## 1. Docstrings

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

## 2. Type Hinting

Always use type hints for function arguments and return values. This improves readability and allows for better static analysis.

## 3. Inline Comments

- Use inline comments sparingly.
- Explain **why** something is done, not **how**. The code should be clear enough to explain the "how".
- Avoid obvious comments like `x = x + 1  # Increment x`.

## 4. TODOs

Use `TODO(username): description` for temporary notes or planned improvements.