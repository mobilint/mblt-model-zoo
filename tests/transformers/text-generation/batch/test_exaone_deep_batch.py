"""Batch tests for EXAONE Deep models."""

import pytest

MODEL_PATHS = (
    "mobilint/EXAONE-Deep-2.4B-Batch16",
    "mobilint/EXAONE-Deep-7.8B-Batch16",
)


@pytest.mark.timeout(300)
def test_exaone_deep_batch(pipe, run_batch_generation) -> None:
    """Run short math prompts against EXAONE Deep batch models."""
    messages = [
        [{"role": "user", "content": "What is 17 + 28?"}],
        [{"role": "user", "content": "What is 96 / 12?"}],
        [{"role": "user", "content": "Solve for x: 3x + 5 = 20."}],
        [{"role": "user", "content": "What is the derivative of x^2?"}],
        [{"role": "user", "content": "Integrate 2x with respect to x."}],
        [{"role": "user", "content": "What is the area of a rectangle with sides 4 and 9?"}],
        [{"role": "user", "content": "What is the next number: 2, 4, 8, 16, ?"}],
        [{"role": "user", "content": "What is 15 percent of 200?"}],
        [{"role": "user", "content": "Simplify: 2(a + 3) - 4."}],
        [{"role": "user", "content": "If y = 5x and x = 7, what is y?"}],
        [{"role": "user", "content": "What is the square root of 144?"}],
        [{"role": "user", "content": "Solve for y: y - 9 = 13."}],
        [{"role": "user", "content": "What is 7 squared?"}],
        [{"role": "user", "content": "A triangle has base 10 and height 6. What is its area?"}],
        [{"role": "user", "content": "Convert 0.75 to a fraction."}],
        [{"role": "user", "content": "What is the mean of 3, 5, and 10?"}],
    ]
    run_batch_generation(pipe, messages)
