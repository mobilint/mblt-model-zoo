"""Model Zoo without ``melotts_mblt``: the facade and CLI bridges must degrade to an install hint.

Kept separate from the facade tests, which skip when the optional extra is not installed, so this runs in every
environment (it blocks the ``melotts_mblt`` import itself).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_WITHOUT_MELOTTS_MBLT = textwrap.dedent(
    """
    import importlib
    import importlib.abc
    import sys


    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "melotts_mblt" or fullname.startswith("melotts_mblt."):
                raise ModuleNotFoundError(f"No module named {fullname!r}", name="melotts_mblt")
            return None


    sys.meta_path.insert(0, _Block())

    import mblt_model_zoo

    assert "MeloTTS" not in mblt_model_zoo.__all__, mblt_model_zoo.__all__
    try:
        import mblt_model_zoo.MeloTTS  # noqa: F401
    except ModuleNotFoundError as exc:
        assert "mblt-model-zoo[MeloTTS]" in str(exc), exc
    else:
        raise AssertionError("facade imported without melotts-mblt")

    from mblt_model_zoo.utils import melotts_download

    assert melotts_download.main() == 2
    cli_main = importlib.import_module("mblt_model_zoo.cli.main")
    commands = (
        ["mblt-model-zoo", "melo", "x", "y.wav"],
        ["mblt-model-zoo", "melotts", "--help"],
        ["mblt-model-zoo", "melo-ui"],
    )
    for argv in commands:
        sys.argv = argv
        try:
            code = cli_main.main()
        except SystemExit as exc:
            code = exc.code
        assert code == 2, (argv, code)
    print("ok")
    """
)


def test_model_zoo_degrades_gracefully_without_melotts_mblt() -> None:
    """Without the extra, Model Zoo imports and its MeloTTS commands report how to install melotts-mblt."""
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_MELOTTS_MBLT], capture_output=True, text=True, check=False, timeout=300
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().splitlines()[-1] == "ok"
    assert "mblt-model-zoo[MeloTTS]" in result.stderr
