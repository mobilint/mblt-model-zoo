from pprint import pprint

import pytest
from transformers import pipeline

MODEL_PATHS_AND_PROMPTS = (
    ("mobilint/bert-base-uncased", "Hello I'm a [MASK] model."),
    ("mobilint/bert-kor-base", "안녕하세요 저는 [MASK] 모델입니다."),
)


@pytest.fixture(params=MODEL_PATHS_AND_PROMPTS, scope="module")
def pipe_and_prompt(request, revision, npu_params):
    model_path, prompt = request.param
    npu_params.warn_unused({"base"})
    model_kwargs = npu_params.base

    if model_kwargs:
        pipe = pipeline(
            "fill-mask",
            model=model_path,
            trust_remote_code=True,
            revision=revision,
            model_kwargs=model_kwargs,
        )
    else:
        pipe = pipeline(
            "fill-mask",
            model=model_path,
            trust_remote_code=True,
            revision=revision,
        )
    yield pipe, prompt
    del pipe


def test_bert(pipe_and_prompt):
    pipe, prompt = pipe_and_prompt
    output = pipe(prompt)
    pprint(output)
