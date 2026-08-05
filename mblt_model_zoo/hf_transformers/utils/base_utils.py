import os
from typing import Optional, Union

from qbruntime import Cluster, Core
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import logging

logger = logging.get_logger(__name__)

cluster_map = {
    0: Cluster.Cluster0,
    1: Cluster.Cluster1,
}

core_map = {
    0: Core.Core0,
    1: Core.Core1,
    2: Core.Core2,
    3: Core.Core3,
}

class PretrainedOnlyMixin(PreTrainedModel):
    def __init__(self, *args, **kwargs):
        _internal_call = kwargs.pop("_internal_call", False)
        self._assert_internal_call(_internal_call)
        super().__init__(*args, **kwargs)
        self._ensure_transformers_5_runtime_attrs()

    def _pretrained_only_base_init(self, config, *args, **kwargs) -> None:
        """Set up the ``PreTrainedModel`` base without cascading through MRO.

        Composite wrappers that inherit an upstream ``*ForConditionalGeneration``
        would otherwise let ``PretrainedOnlyMixin.__init__``'s ``super().__init__``
        walk into the upstream constructor, which instantiates its own vision/
        text submodules — an MXQ load we immediately throw away when the
        wrapper overwrites ``self.model``. Calling ``PreTrainedModel.__init__``
        directly skips that upstream body.
        """
        _internal_call = kwargs.pop("_internal_call", False)
        self._assert_internal_call(_internal_call)
        PreTrainedModel.__init__(self, config, *args, **kwargs)
        self._ensure_transformers_5_runtime_attrs()

    def _assert_internal_call(self, _internal_call: bool) -> None:
        if _internal_call:
            return
        cls_name = self.__class__.__name__
        raise RuntimeError(
            f"Direct instantiation of {cls_name} is not allowed.\n"
            f"Please use `{cls_name}.from_pretrained(...)` to load the NPU model correctly."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        kwargs["_internal_call"] = True
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    def _ensure_transformers_5_runtime_attrs(self) -> None:
        """Populate runtime attrs expected by newer Transformers releases."""
        if getattr(self, "_tp_plan", None) is None:
            self._tp_plan = {}
        if getattr(self, "_ep_plan", None) is None:
            self._ep_plan = {}
        if getattr(self, "_pp_plan", None) is None:
            self._pp_plan = {}
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
