"""Deprecation notices that point Model Zoo users to the standalone Mobilint packages.

Each check runs in a fresh interpreter because the notices fire once, at module import or CLI start.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from mblt_model_zoo._deprecation import SUPPRESS_ENV_VAR


def _run(code: str, *, suppress: bool = False) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != SUPPRESS_ENV_VAR}
    if suppress:
        env[SUPPRESS_ENV_VAR] = "1"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )


def test_bare_package_import_is_silent_and_lazy() -> None:
    result = _run(
        """
        import sys
        import warnings

        warnings.simplefilter("error")
        import mblt_model_zoo

        assert mblt_model_zoo.__version__
        assert "mblt_model_zoo.vision" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("attribute", "package"), [("hf_transformers", "transformers_mblt"), ("MeloTTS", "melotts_mblt")]
)
def test_missing_optional_subpackage_is_absent_attribute(attribute: str, package: str) -> None:
    result = _run(
        f"""
        import importlib.abc
        import inspect
        import sys


        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == {package!r} or fullname.startswith({package!r} + "."):
                    raise ModuleNotFoundError(f"No module named {{fullname!r}}", name={package!r})
                return None


        sys.meta_path.insert(0, _Block())

        import mblt_model_zoo

        assert not hasattr(mblt_model_zoo, {attribute!r})
        assert getattr(mblt_model_zoo, {attribute!r}, None) is None
        assert {attribute!r} not in dir(mblt_model_zoo)
        inspect.getmembers(mblt_model_zoo)
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("module", "package", "replacement"),
    [
        ("mblt_model_zoo.vision", "mblt-vision-python", "mblt_vision"),
        ("mblt_model_zoo.compile", "mblt-vision-python", "mblt_vision.compile"),
        ("mblt_model_zoo.utils.npu_backend", "mblt-npu-python", "mblt_npu"),
        ("mblt_model_zoo.utils.npu_target", "mblt-npu-python", "mblt_npu.npu_target"),
        ("mblt_model_zoo.utils.logging", "mblt-npu-python", "mblt_npu"),
    ],
)
def test_facade_import_warns_with_replacement(module: str, package: str, replacement: str) -> None:
    result = _run(
        f"""
        import importlib
        import warnings

        from mblt_model_zoo._deprecation import MbltModelZooDeprecationWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module({module!r})
        messages = [str(w.message) for w in caught if issubclass(w.category, MbltModelZooDeprecationWarning)]
        assert issubclass(MbltModelZooDeprecationWarning, FutureWarning)
        assert len(messages) == 1, messages
        assert {package!r} in messages[0] and "import {replacement} instead" in messages[0], messages[0]
        """
    )
    assert result.returncode == 0, result.stderr


def test_facade_warning_is_shown_by_default_filters() -> None:
    result = _run("from mblt_model_zoo.vision import MBLT_Engine")
    assert result.returncode == 0, result.stderr
    assert "MbltModelZooDeprecationWarning" in result.stderr
    assert "pip install mblt-vision-python" in result.stderr


def test_environment_variable_suppresses_warnings() -> None:
    result = _run(
        """
        import warnings

        warnings.simplefilter("error")
        import mblt_model_zoo.vision
        import mblt_model_zoo.compile
        """,
        suppress=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["predict", "-h"], "`mblt-vision predict` from mblt-vision-python"),
        (["val", "-h"], "`mblt-vision val` from mblt-vision-python"),
        (["compile", "-h"], "`mblt-vision compile` from mblt-vision-python"),
        (["melo-ui", "-h"], "`melotts-mblt ui` from melotts-mblt"),
        (["-h"], "mblt-vision-python (`mblt-vision`), transformers-mblt"),
    ],
)
def test_cli_prints_one_notice_naming_replacement(argv: list[str], expected: str) -> None:
    result = _run(f"import sys; sys.argv = ['mblt-model-zoo', *{argv!r}]; from mblt_model_zoo.cli import main; main()")
    assert result.returncode == 0, result.stderr
    assert expected in result.stderr
    assert result.stderr.count("is deprecated") == 1, result.stderr


def test_cli_notice_is_suppressible() -> None:
    result = _run(
        "import sys; sys.argv = ['mblt-model-zoo', 'predict', '-h']; from mblt_model_zoo.cli import main; main()",
        suppress=True,
    )
    assert result.returncode == 0, result.stderr
    assert "deprecated" not in result.stderr
