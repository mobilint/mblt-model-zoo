"""Regression tests for runtime compatibility with newer Transformers releases."""

from __future__ import annotations

from mblt_model_zoo.hf_transformers.utils import base_utils


class _DummyPretrainedOnlyModel(base_utils.PretrainedOnlyMixin):
    """Minimal model used to exercise PretrainedOnlyMixin initialization."""

    _tp_plan = None
    _ep_plan = None
    _pp_plan = None

    def __init__(self, config: object, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)


def test_pretrained_only_mixin_sets_transformers_5_runtime_attrs(monkeypatch) -> None:
    """Populate runtime attrs expected by Transformers 5.x when subclasses skip post_init."""

    def _fake_pretrained_model_init(self, config: object, *args, **kwargs) -> None:
        self.config = config

    monkeypatch.setattr(base_utils.PreTrainedModel, "__init__", _fake_pretrained_model_init)

    model = _DummyPretrainedOnlyModel(object(), _internal_call=True)

    assert model.all_tied_weights_keys == {}
    assert model._tp_plan == {}
    assert model._ep_plan == {}
    assert model._pp_plan == {}
