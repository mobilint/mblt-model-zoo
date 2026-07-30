"""Shared fixtures for batch image-text-to-text tests."""

from __future__ import annotations

from typing import Optional

import pytest
from transformers import AutoProcessor, pipeline

from tests.npu_backend_options import (
    VisionTextNpuParams,
    collect_npu_kwargs,
    option_value_was_provided,
    validate_single_only_core_mode,
)
from tests.transformers.text_generation.utils import BatchTextStreamer


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize the shared batch pipeline fixture from module-level model paths."""
    if "pipe" not in metafunc.fixturenames:
        return

    model_paths = getattr(metafunc.module, "MODEL_PATHS", None)
    if not model_paths:
        return

    metafunc.parametrize("pipe", model_paths, indirect=True, ids=list(model_paths), scope="module")


@pytest.fixture(scope="module")
def vision_text_npu_params(
    request: pytest.FixtureRequest,
    embedding_weight: Optional[str],
) -> VisionTextNpuParams:
    """Return vision/text backend kwargs for batch VLM suites.

    Batched text MXQ builds are compiled with single-core mode only, so the
    text backend is forced accordingly. Vision-side settings remain whatever
    the CLI supplied.
    """
    validate_single_only_core_mode(request.config, suite_name="Batch image-text-to-text tests")

    vision_kwargs, _ = collect_npu_kwargs(request.config, "vision")
    text_kwargs, _ = collect_npu_kwargs(
        request.config,
        "text",
        core_mode_override="single",
    )
    # The compiled batched text MXQ only supports a specific single core
    # (`0:0`), so drop stray `text_target_cores` unless the CLI provided one.
    if not option_value_was_provided(request.config, "text", "target_cores"):
        text_kwargs.pop("text_target_cores", None)

    return VisionTextNpuParams(vision=vision_kwargs, text=text_kwargs)


@pytest.fixture(scope="module")
def pipe(
    request: pytest.FixtureRequest,
    revision: Optional[str],
    vision_text_npu_params: VisionTextNpuParams,
):
    """Create a batch-capable image-text-to-text pipeline for the parametrized model."""
    model_path = request.param
    model_kwargs = {**vision_text_npu_params.vision, **vision_text_npu_params.text}

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision=revision,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    pipe = pipeline(
        "image-text-to-text",
        model=model_path,
        processor=processor,
        trust_remote_code=True,
        revision=revision,
        model_kwargs=model_kwargs or None,
    )

    yield pipe
    del pipe


@pytest.fixture
def run_batch_generation(batch_generation_token_limit: int):
    """Run generation for a list of batched chat conversations with a shared cap."""

    def _run(
        pipe,
        conversations: list[list[dict]],
        max_new_tokens: Optional[int] = None,
    ) -> None:
        pipe.generation_config.max_new_tokens = None
        pipe.generation_config.max_length = None
        batch_size = len(conversations)
        pipe(
            text=conversations,
            batch_size=batch_size,
            generate_kwargs={
                "max_new_tokens": max_new_tokens if max_new_tokens is not None else batch_generation_token_limit,
                "streamer": BatchTextStreamer(
                    tokenizer=pipe.tokenizer,
                    batch_size=batch_size,
                    skip_prompt=False,
                ),
            },
        )

    return _run
