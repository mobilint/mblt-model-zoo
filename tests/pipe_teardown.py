"""Shared teardown helpers for NPU-backed ``pipeline(...)`` fixtures."""

from __future__ import annotations

import functools
import gc
from typing import Any, Callable

import pytest


def release_pipe(pipe: Any) -> None:
    """Dispose the NPU backend held by ``pipe``, drop the reference, force GC.

    Relying on ``del pipe`` alone leaves ``model.dispose()`` waiting on Python
    to collect the object. On Windows the driver has not always reclaimed
    LPDDR by the next module's ``pipeline(...)`` call, producing intermittent
    :class:`MobilintBackendAllocError`. Calling ``dispose()`` synchronously
    makes the release ordered against the next allocation.
    """
    model = getattr(pipe, "model", None)
    if model is not None and hasattr(model, "dispose"):
        model.dispose()
    elif hasattr(pipe, "dispose"):
        pipe.dispose()
    del pipe
    gc.collect()


def pipe_fixture(*, scope: str = "module", **fixture_kwargs: Any):
    """Wrap a ``pipe`` builder into a fixture with shared NPU teardown.

    Each suite declares ``def pipe(...) -> Pipeline: return pipeline(...)`` with
    whatever fixture dependencies it needs; the decorator turns it into a
    ``@pytest.fixture(scope=scope, **fixture_kwargs)`` generator that yields
    the built pipeline and calls :func:`release_pipe` in a ``finally`` block.
    Keeps the dispose + GC contract in one place so future teardown tweaks do
    not need to be fanned out across every suite's conftest.

    ``fixture_kwargs`` forwards any extra ``pytest.fixture`` arguments (e.g.
    ``params=`` for the MeloTTS pipe that uses the built-in fixture-params
    hook rather than :func:`pytest_generate_tests`).
    """

    def deco(build_fn: Callable[..., Any]):
        @pytest.fixture(scope=scope, **fixture_kwargs)
        @functools.wraps(build_fn)
        def wrapper(*args, **kwargs):
            pipe = build_fn(*args, **kwargs)
            try:
                yield pipe
            finally:
                release_pipe(pipe)

        return wrapper

    return deco
