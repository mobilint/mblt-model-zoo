"""Compatibility checks for the Model Zoo MeloTTS facade over :mod:`melotts_mblt`."""

from __future__ import annotations

import importlib
import importlib.resources
import sys

import pytest

melotts_mblt = pytest.importorskip("melotts_mblt")

LEGACY = "mblt_model_zoo.MeloTTS"


# ``app`` is deliberately absent: importing it builds both TTS models at module import time.
@pytest.mark.parametrize(
    "path", [".api", ".models", ".utils", ".commons", ".download_utils", ".text", ".text.english", ".text.korean"]
)
def test_legacy_module_paths_are_standalone_module_objects(path: str) -> None:
    assert importlib.import_module(LEGACY + path) is importlib.import_module("melotts_mblt" + path)


def test_legacy_names_are_standalone_objects() -> None:
    import mblt_model_zoo.MeloTTS as facade
    from mblt_model_zoo.MeloTTS.api import TTS

    assert TTS is melotts_mblt.TTS
    assert facade.TTS is melotts_mblt.TTS
    assert facade.__name__ == LEGACY


def test_packaged_text_resources_resolve_through_the_alias() -> None:
    """The alias keeps the standalone ``__spec__`` so ``importlib.resources`` finds the packaged ``cmudict``."""
    legacy_text = importlib.import_module(f"{LEGACY}.text")
    assert legacy_text.__spec__.name == "melotts_mblt.text"
    assert importlib.resources.files(legacy_text).joinpath("cmudict.rep").is_file()


def test_unknown_legacy_submodule_raises_module_not_found() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{LEGACY}.not_a_real_module")


def test_melo_commands_forward_to_standalone_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(
        "melotts_mblt.cli.tts.run_tts", lambda args, prog_name: calls.append((list(args), prog_name)) or 0
    )
    main = importlib.import_module("mblt_model_zoo.cli.main")  # the package re-exports the main() function

    for command in ("melo", "melotts"):
        monkeypatch.setattr(sys, "argv", ["mblt-model-zoo", command, "Hello", "out.wav", "--language", "KR"])
        assert main.main() == 0
    assert calls == [
        (["Hello", "out.wav", "--language", "KR"], "mblt-model-zoo melo"),
        (["Hello", "out.wav", "--language", "KR"], "mblt-model-zoo melotts"),
    ]


def test_melo_ui_forwards_to_standalone_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("melotts_mblt.cli.ui.run_ui", lambda **kwargs: calls.append(kwargs) or 0)
    main = importlib.import_module("mblt_model_zoo.cli.main")
    monkeypatch.setattr(sys, "argv", ["mblt-model-zoo", "melo-ui", "--share", "--host", "0.0.0.0", "--port", "7860"])

    assert main.main() == 0
    assert calls == [{"share": True, "host": "0.0.0.0", "port": 7860}]


def test_melo_ui_short_flags_match_standalone_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    """``-s`` / ``-p`` work as in ``melotts-mblt ui`` (and the upstream ``melo-ui`` Click command)."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("melotts_mblt.cli.ui.run_ui", lambda **kwargs: calls.append(kwargs) or 0)
    main = importlib.import_module("mblt_model_zoo.cli.main")
    monkeypatch.setattr(sys, "argv", ["mblt-model-zoo", "melo-ui", "-s", "-p", "7861"])

    assert main.main() == 0
    assert calls == [{"share": True, "host": None, "port": 7861}]


def test_melotts_download_script_forwards_to_standalone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("melotts_mblt.cli.download.run_download", lambda: 0)
    from mblt_model_zoo.utils import melotts_download

    assert melotts_download.main() == 0
