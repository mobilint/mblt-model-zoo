"""Compatibility checks for the Model Zoo Transformers facade over :mod:`transformers_mblt`."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import textwrap

import pytest

transformers_mblt = pytest.importorskip("transformers_mblt")

LEGACY = "mblt_model_zoo.hf_transformers"


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".models",
        ".models.llama",
        ".models.llama.configuration_llama",
        ".models.llama.modeling_llama",
        ".models.whisper.processing_whisper",
        ".models.bert.proxy_bert",
        ".utils",
        ".utils.api",
        ".utils.cache_utils",
        ".utils.modeling_utils",
        ".utils.eagle3.tree_decoding",
    ],
)
def test_legacy_module_paths_are_standalone_module_objects(path: str) -> None:
    """Every legacy import path must resolve to the same module object, never a copied implementation."""
    legacy = importlib.import_module(LEGACY + path)
    if not path:
        assert legacy.__name__ == LEGACY
        return
    assert legacy is importlib.import_module("transformers_mblt" + path)


@pytest.mark.parametrize("path", [".models.llama", ".models.llama.modeling_llama", ".utils.eagle3"])
def test_legacy_import_keeps_standalone_module_spec(path: str) -> None:
    """A legacy import must not replace the standalone module's ``__spec__`` with the alias spec."""
    standalone = importlib.import_module("transformers_mblt" + path)
    is_package = hasattr(standalone, "__path__")
    importlib.import_module(LEGACY + path)
    assert standalone.__spec__.name == "transformers_mblt" + path
    assert (standalone.__spec__.submodule_search_locations is not None) is is_package
    assert importlib.util.find_spec("transformers_mblt" + path).name == "transformers_mblt" + path


def test_legacy_names_are_standalone_objects() -> None:
    from transformers_mblt.models.llama.modeling_llama import MobilintLlamaForCausalLM as Standalone

    import mblt_model_zoo.hf_transformers as facade
    from mblt_model_zoo.hf_transformers.models.llama.modeling_llama import MobilintLlamaForCausalLM
    from mblt_model_zoo.hf_transformers.utils import list_models, list_tasks

    assert MobilintLlamaForCausalLM is Standalone
    assert list_models is transformers_mblt.list_models
    assert list_tasks is transformers_mblt.list_tasks
    assert facade.register is transformers_mblt.register
    assert facade.models is transformers_mblt.models


def test_hub_proxy_style_legacy_import_registers_standalone_classes() -> None:
    """Hub `proxy_*.py` files import the legacy path; they must receive the standalone, Auto-registered classes."""
    namespace: dict[str, object] = {}
    exec(  # noqa: S102 - mirrors the remote-code proxy published on the Hub
        "from mblt_model_zoo.hf_transformers.models.llama.configuration_llama import MobilintLlamaConfig\n"
        "from mblt_model_zoo.hf_transformers.models.llama.modeling_llama import MobilintLlamaForCausalLM\n",
        namespace,
    )
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    assert namespace["MobilintLlamaConfig"] is CONFIG_MAPPING["mobilint-llama"]
    assert namespace["MobilintLlamaForCausalLM"].__module__ == "transformers_mblt.models.llama.modeling_llama"


def test_unknown_legacy_submodule_raises_module_not_found() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{LEGACY}.models.not_a_real_arch")


def test_cli_modules_are_standalone_aliases() -> None:
    for name in ("tps", "tps_table", "chat", "transformers_compat"):
        assert importlib.import_module(f"mblt_model_zoo.cli.{name}") is importlib.import_module(
            f"transformers_mblt.cli.{name}"
        )


def test_model_zoo_cli_exposes_standalone_tps_parser() -> None:
    main = importlib.import_module("mblt_model_zoo.cli.main")
    args = main.build_parser().parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct"])
    assert args.model == "mobilint/Llama-3.2-1B-Instruct"
    assert main.is_transformers_cli_command(["mblt-model-zoo", "chat"])


def test_npu_backend_dispatcher_uses_standalone_multi_slot_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Model Zoo ``dispatcher`` fallback builds the transformers-mblt dispatcher and caches it per backend."""
    import transformers_mblt.utils.multi_slot_dispatch as dispatch_module

    from mblt_model_zoo.utils import npu_backend

    class _StubDispatcher:
        def __init__(self, backend: object) -> None:
            self.backend = backend

    monkeypatch.setattr(dispatch_module, "MultiSlotDispatcher", _StubDispatcher)
    backend = type("_Backend", (), {})()

    dispatcher = npu_backend._get_transformers_dispatcher(backend)

    assert isinstance(dispatcher, _StubDispatcher)
    assert dispatcher.backend is backend
    assert npu_backend._get_transformers_dispatcher(backend) is dispatcher
    assert isinstance(npu_backend.MobilintNPUBackend.__dict__.get("dispatcher"), property)


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
