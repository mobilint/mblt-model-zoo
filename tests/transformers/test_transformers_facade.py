"""Compatibility checks for the Model Zoo Transformers facade over :mod:`transformers_mblt`."""

from __future__ import annotations

import importlib
import importlib.util

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
