"""Core-mode validation contracts."""

from mblt_model_zoo.utils.core_mode import normalize_core_mode


def test_normalize_core_mode_accepts_auto() -> None:
    """``auto`` is a valid runtime mode for per-layer scheduled MXQs."""
    assert normalize_core_mode("auto") == "auto"
