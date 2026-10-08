"""Deprecation notices that point Model Zoo users to the standalone Mobilint packages.

``mblt-model-zoo`` is deprecated and no longer receives feature updates. Its compatibility facades and CLI keep
working, but every entry point tells the user which standalone package replaces it:

- Vision and compilation: ``mblt-vision-python`` (``import mblt_vision``, ``mblt-vision`` CLI).
- Hugging Face Transformers: ``transformers-mblt`` (``import transformers_mblt``, ``transformers-mblt`` CLI).
- MeloTTS: ``melotts-mblt`` (``import melotts_mblt``, ``melotts-mblt`` CLI).
- NPU backend utilities: ``mblt-npu-python`` (``import mblt_npu``).

Python facades emit :class:`FutureWarning`, which Python shows to end users by default, unlike
:class:`DeprecationWarning`. CLI commands print one notice to stderr. Set ``MBLT_MODEL_ZOO_SUPPRESS_DEPRECATION=1``
to silence both, or filter the warning with the standard :mod:`warnings` machinery.
"""

from __future__ import annotations

import os
import sys
import warnings

SUPPRESS_ENV_VAR = "MBLT_MODEL_ZOO_SUPPRESS_DEPRECATION"
MIGRATION_URL = "https://github.com/mobilint/mblt-model-zoo#deprecation-notice"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_cli_notice_shown = False


class MbltModelZooDeprecationWarning(FutureWarning):
    """Warning category for deprecated ``mblt_model_zoo`` import paths."""


def is_suppressed() -> bool:
    """Return whether the user silenced deprecation notices through :data:`SUPPRESS_ENV_VAR`."""
    return os.environ.get(SUPPRESS_ENV_VAR, "").strip().lower() in _TRUTHY


def warn_deprecated_import(legacy: str, package: str, replacement: str, stacklevel: int = 3) -> None:
    """Warn that a legacy ``mblt_model_zoo`` import path is deprecated.

    The warning is skipped when it is suppressed or when the Model Zoo CLI already printed its notice in this
    process.

    Args:
        legacy: Deprecated import path, for example ``"mblt_model_zoo.vision"``.
        package: PyPI distribution that replaces it, for example ``"mblt-vision-python"``.
        replacement: Import path to use instead, for example ``"mblt_vision"``.
        stacklevel: Frame the warning is attributed to. The default points at the module that imported the
            facade calling this helper; import machinery frames are skipped by :mod:`warnings`.
    """
    if _cli_notice_shown or is_suppressed():
        return
    warnings.warn(
        f"{legacy} is deprecated: mblt-model-zoo no longer receives updates. "
        f"Install {package} (pip install {package}) and import {replacement} instead. "
        f"Migration guide: {MIGRATION_URL}",
        MbltModelZooDeprecationWarning,
        stacklevel=stacklevel,
    )


def print_cli_notice(command: str | None, package: str | None, replacement: str | None) -> None:
    """Print the Model Zoo CLI deprecation notice to stderr once per process.

    Args:
        command: Legacy invocation, for example ``"mblt-model-zoo predict"``; ``None`` for the bare command.
        package: PyPI distribution that replaces it, or ``None`` when the command has no single replacement.
        replacement: Command to run instead, for example ``"mblt-vision predict"``.
    """
    global _cli_notice_shown
    if _cli_notice_shown:
        return
    _cli_notice_shown = True
    if is_suppressed():
        return
    if command and package and replacement:
        guidance = f"Use `{replacement}` from {package} instead (pip install {package})."
    else:
        guidance = (
            "Use the standalone packages instead: mblt-vision-python (`mblt-vision`), "
            "transformers-mblt (`transformers-mblt`), and melotts-mblt (`melotts-mblt`)."
        )
    subject = f"`{command}`" if command else "mblt-model-zoo"
    print(
        f"DeprecationWarning: {subject} is deprecated; mblt-model-zoo no longer receives updates. {guidance} "
        f"Migration guide: {MIGRATION_URL} (set {SUPPRESS_ENV_VAR}=1 to hide this notice)",
        file=sys.stderr,
    )
