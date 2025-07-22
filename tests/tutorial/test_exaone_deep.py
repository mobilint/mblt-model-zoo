import torch
from mblt_model_zoo.transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread

model_name = "mobilint/EXAONE-Deep-2.4B"
streaming = True  # choose the streaming option

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)

if streaming:
    streamer = TextIteratorStreamer(tokenizer)
    thread = Thread(
        target=model.generate,
        kwargs=dict(
            input_ids=input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            streamer=streamer,
        ),
    )
    thread.start()

    for text in streamer:
        print(text, end="", flush=True)
else:
    output = model.generate(
        input_ids,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
    )
    print(tokenizer.decode(output[0]))
