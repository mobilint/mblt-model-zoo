from pprint import pprint

import pytest

from mblt_model_zoo.transformers import pipeline


@pytest.fixture
def pipe():
    model_name = "mobilint/bert-base-uncased"
    pipe = pipeline("fill-mask", model=model_name)
    yield pipe
    pipe.model.dispose()


def test_bert(pipe):
    output = pipe("Hello I'm a [MASK] model.")
    pprint(output)
