import pytest
from transformers import TextStreamer

from mblt_model_zoo.transformers import AutoTokenizer, pipeline

MODEL_PATHS = (
    "mobilint/HyperCLOVAX-SEED-Text-Instruct-0.5B",
    "mobilint/HyperCLOVAX-SEED-Text-Instruct-1.5B",
)


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


def test_clova(pipe):
    pipe.generation_config.max_new_tokens = None

    messages = [
        {"role": "tool_list", "content": ""},
        {
            "role": "system",
            "content": '- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n- 오늘은 2025년 04월 24일(목)이다.',
        },
        {
            "role": "user",
            "content": "슈뢰딩거 방정식과 양자역학의 관계를 최대한 자세히 알려줘.",
        },
    ]

    outputs = pipe(
        messages,
        max_length=512,
    )
