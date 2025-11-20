import pytest
from transformers import TextStreamer

from mblt_model_zoo.transformers import AutoTokenizer, pipeline

MODEL_PATHS = ("mobilint/EXAONE-Deep-2.4B",)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, mxq_path):
    model_path = request.param

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if mxq_path:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            model_kwargs={"mxq_path": mxq_path},
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
        )
    yield pipe
    pipe.model.dispose()


@pytest.mark.timeout(300)
def test_exaone_deep(pipe):
    pipe.generation_config.max_new_tokens = None

    # Choose your prompt:
    #   Math example (AIME 2024)
    prompt = r"""Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
\[\log_2\left({x \over yz}\right) = {1 \over 2}\]\[\log_2\left({y \over xz}\right) = {1 \over 3}\]\[\log_2\left({z \over xy}\right) = {1 \over 4}\]
Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.

Please reason step by step, and put your final answer within \boxed{}."""
    #   Korean MCQA example (CSAT Math 2025)
    prompt = r"""Question : $a_1 = 2$인 수열 $\{a_n\}$과 $b_1 = 2$인 등차수열 $\{b_n\}$이 모든 자연수 $n$에 대하여\[\sum_{k=1}^{n} \frac{a_k}{b_{k+1}} = \frac{1}{2} n^2\]을 만족시킬 때, $\sum_{k=1}^{5} a_k$의 값을 구하여라.

Options :
A) 120
B) 125
C) 130
D) 135
E) 140
 
Please reason step by step, and you should write the correct option alphabet (A, B, C, D or E) within \\boxed{}."""

    messages = [{"role": "user", "content": prompt}]

    outputs = pipe(
        messages,
        max_length=512,
    )
