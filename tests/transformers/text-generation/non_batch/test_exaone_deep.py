"""Non-batch tests for EXAONE Deep models."""

MODEL_PATHS = (
    "mobilint/EXAONE-Deep-2.4B",
    "mobilint/EXAONE-Deep-7.8B",
)


def test_exaone_deep(pipe, generation_token_limit: int) -> None:
    """Run a math-style prompt against EXAONE Deep."""
    pipe.generation_config.max_new_tokens = None
    pipe.generation_config.max_length = None

    prompt = (
        r"Question: Let $\{a_n\}$ be a sequence with $a_1 = 2$ and let $\{b_n\}$ "
        r"be a geometric sequence with $b_1 = 2$."
        r" Suppose that for every positive integer $n$,"
        r" \[\sum_{k=1}^{n} \frac{a_k}{b_{k+1}} = \frac{1}{2} n^2\]"
        r" Find the value of $\sum_{k=1}^{5} a_k$."
        "\n\n"
        "Options:"
        "\nA) 120\nB) 125\nC) 130\nD) 135\nE) 140\n\n"
        "Please reason step by step, and you should write the correct option alphabet "
        r"(A, B, C, D or E) within \\boxed{}."
    )

    messages = [{"role": "user", "content": prompt}]

    pipe(messages, max_new_tokens=generation_token_limit)
