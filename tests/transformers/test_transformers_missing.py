"""Model Zoo without ``transformers_mblt``: the facade and CLI bridges must degrade to an install hint.

Kept separate from the facade tests, which skip when the optional extra is not installed, so this runs in every
environment (it blocks the ``transformers_mblt`` import itself).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_WITHOUT_TRANSFORMERS_MBLT = textwrap.dedent(
    """
    import importlib
    import importlib.abc
    import sys


    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "transformers_mblt" or fullname.startswith("transformers_mblt."):
                raise ModuleNotFoundError(f"No module named {fullname!r}", name="transformers_mblt")
            return None


    sys.meta_path.insert(0, _Block())

    import mblt_model_zoo

    assert "hf_transformers" not in mblt_model_zoo.__all__, mblt_model_zoo.__all__
    try:
        import mblt_model_zoo.hf_transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        assert "mblt-model-zoo[transformers]" in str(exc), exc
    else:
        raise AssertionError("facade imported without transformers-mblt")

    import runpy

    bridges = ("tps", "tps_table", "chat", "transformers_compat")
    for bridge in bridges:
        module = importlib.import_module(f"mblt_model_zoo.cli.{bridge}")  # must import without transformers-mblt
        assert not hasattr(module, "__path__")  # dunder lookups stay AttributeError-safe
    for bridge, name in (
        ("chat", "register_mobilint_models"),
        ("tps_table", "TPS_TABLE_ROWS"),
        ("tps", "Eagle3PipelineOptions"),
        ("transformers_compat", "_prepare_transformers_cli"),
    ):
        try:
            exec(f"from mblt_model_zoo.cli.{bridge} import {name}")
        except ModuleNotFoundError as exc:
            assert "mblt-model-zoo[transformers]" in str(exc), exc
        else:
            raise AssertionError(f"{bridge}.{name} resolved without transformers-mblt")
    for bridge in bridges:
        try:
            runpy.run_module(f"mblt_model_zoo.cli.{bridge}", run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 2, (bridge, exc.code)
        else:
            raise AssertionError(f"running {bridge} did not exit")

    # The placeholders cli.main relies on stay real attributes rather than going through the fallback.
    assert callable(importlib.import_module("mblt_model_zoo.cli.tps").add_tps_parser)
    compat = importlib.import_module("mblt_model_zoo.cli.transformers_compat")
    assert compat.is_transformers_cli_command(["mblt-model-zoo", "chat"])

    cli_main = importlib.import_module("mblt_model_zoo.cli.main")  # the package re-exports the main() function

    for argv in (["mblt-model-zoo", "tps", "measure", "--model", "x"], ["mblt-model-zoo", "chat", "x"]):
        sys.argv = argv
        try:
            code = cli_main.main()
        except SystemExit as exc:
            code = exc.code
        assert code == 2, (argv, code)
    print("ok")
    """
)


def test_model_zoo_degrades_gracefully_without_transformers_mblt() -> None:
    """Without the extra, Model Zoo imports and its CLI reports how to install transformers-mblt."""
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_TRANSFORMERS_MBLT], capture_output=True, text=True, check=False, timeout=300
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().splitlines()[-1] == "ok"
    assert "mblt-model-zoo[transformers]" in result.stderr
