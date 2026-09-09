"""Shared teardown helpers for NPU-backed ``pipeline(...)`` fixtures."""

from __future__ import annotations

import functools
import gc
from typing import Any, Callable

import pytest


def _try_dispose(target: Any) -> None:
    """Call ``target.dispose()`` if it exists; swallow failures on the release path."""
    dispose = getattr(target, "dispose", None)
    if not callable(dispose):
        return
    try:
        dispose()
    except Exception:  # noqa: BLE001 — teardown must never raise
        pass


def release_pipe(pipe: Any) -> None:
    """Dispose every NPU-backing submodule reachable from ``pipe``.

    Composite HF pipelines (Qwen3-VL, Whisper, Qwen3-ASR) wrap the actual
    NPU-loaded backends inside nested modules (``model.visual`` +
    ``model.language_model``, ``model.encoder`` + ``model.decoder``,
    ``thinker.audio_tower``, ...), and the outer ``pipe.model`` conditional-
    generation wrapper has no ``dispose()`` of its own. Walk the
    ``torch.nn.Module`` tree so every leaf that owns an NPU handle is
    released. Non-torch objects (e.g. ``MeloTTS.TTS``) get their own
    top-level ``dispose()`` call as a fallback.

    Callers own the ``del`` + ``gc.collect()`` sequence themselves: this
    helper's ``pipe`` local is not the caller's binding, so any ``del``
    here only drops the helper's reference. :func:`pipe_fixture` — and the
    hand-rolled callers under ``tests/`` — issue ``del`` and ``gc.collect``
    on the caller-scope reference right after this returns so Python can
    reclaim the Python-side objects before the next fixture allocates.
    """
    subject = getattr(pipe, "model", None)
    if subject is None:
        subject = pipe

    modules_attr = getattr(subject, "modules", None)
    if callable(modules_attr):
        for sub in modules_attr():
            _try_dispose(sub)
    else:
        _try_dispose(subject)


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
                del pipe
                gc.collect()

        return wrapper

    return deco
