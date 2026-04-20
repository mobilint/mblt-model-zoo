"""Shared fixtures for batch text-generation tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pytest
from transformers import AutoTokenizer, pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import BatchTextStreamer  # noqa: E402


def _parse_target_cores(value: Optional[str]) -> Optional[list[str]]:
    """Parse a semicolon-delimited target core option."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return [item.strip() for item in text.split(";") if item.strip()]


def _parse_target_clusters(value: Optional[str]) -> Optional[list[int]]:
    """Parse a semicolon-delimited target cluster option."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return [int(item.strip()) for item in text.split(";") if item.strip()]


def _build_model_kwargs(request: pytest.FixtureRequest, embedding_weight: Optional[str]) -> dict[str, Any]:
    """Build model kwargs from shared pytest CLI options."""
    config = request.config
    model_kwargs: dict[str, Any] = {}

    mxq_path = config.getoption("--mxq-path")
    if mxq_path:
        model_kwargs["mxq_path"] = mxq_path

    dev_no = config.getoption("--dev-no")
    if dev_no is not None:
        model_kwargs["dev_no"] = dev_no

    raw_core_mode = config.getoption("--core-mode")
    core_mode = None if raw_core_mode in {None, "", "all"} else raw_core_mode
    if core_mode:
        model_kwargs["core_mode"] = core_mode

    target_cores = _parse_target_cores(config.getoption("--target-cores"))
    if target_cores is not None:
        model_kwargs["target_cores"] = target_cores

    target_clusters = _parse_target_clusters(config.getoption("--target-clusters"))
    if target_clusters is not None:
        model_kwargs["target_clusters"] = target_clusters

    if core_mode == "single" and target_cores is None:
        model_kwargs["target_cores"] = ["0:0"]
    elif core_mode == "global4" and target_clusters is None:
        model_kwargs["target_clusters"] = [0]
    elif core_mode == "global8" and target_clusters is None:
        model_kwargs["target_clusters"] = [0, 1]

    if embedding_weight:
        model_kwargs["embedding_weight"] = embedding_weight

    return model_kwargs


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize the shared batch pipeline fixture from module-level model paths."""
    if "pipe" not in metafunc.fixturenames:
        return

    model_paths = getattr(metafunc.module, "MODEL_PATHS", None)
    if not model_paths:
        return

    metafunc.parametrize("pipe", model_paths, indirect=True, ids=list(model_paths), scope="module")


@pytest.fixture(scope="module")
def pipe(request: pytest.FixtureRequest, revision: Optional[str], embedding_weight: Optional[str]):
    """Create a batch-capable text-generation pipeline for the parametrized model."""
    model_path = request.param
    model_kwargs = _build_model_kwargs(request, embedding_weight)

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if model_kwargs:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            trust_remote_code=True,
            revision=revision,
            model_kwargs=model_kwargs,
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            trust_remote_code=True,
            revision=revision,
        )

    yield pipe
    del pipe


@pytest.fixture
def run_batch_generation():
    """Run generation for a list of batched chat messages."""

    def _run(pipe, messages: list[list[dict[str, str]]], max_new_tokens: int = 256) -> None:
        pipe.generation_config.max_new_tokens = None
        batch_size = len(messages)
        pipe(
            messages,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            streamer=BatchTextStreamer(
                tokenizer=pipe.tokenizer,
                batch_size=batch_size,
                skip_prompt=False,
            ),
        )

    return _run
