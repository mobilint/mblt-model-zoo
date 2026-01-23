from pprint import pprint

import pytest
from transformers import pipeline

MODEL_PATHS_AND_PROMPTS = (
    ("mobilint/bert-base-uncased", "Hello I'm a [MASK] model."),
    ("mobilint/bert-kor-base", "안녕하세요 저는 [MASK] 모델입니다."),
)


@pytest.fixture(params=MODEL_PATHS_AND_PROMPTS, scope="module")
def pipe_and_prompt(request, mxq_path):
    model_path, prompt = request.param

    if mxq_path:
        pipe = pipeline(
            "fill-mask",
            model=model_path,
            trust_remote_code=True,
            model_kwargs={"mxq_path": mxq_path},
        )
    else:
        pipe = pipeline(
            "fill-mask",
            model=model_path,
            trust_remote_code=True,
        )
    yield pipe, prompt
    if isinstance(pipe.model.bert, MobilintModelMixin):
        pipe.model.bert.dispose()


def test_bert(pipe_and_prompt):
    pipe, prompt = pipe_and_prompt
    output = pipe(prompt)
    pprint(output)
