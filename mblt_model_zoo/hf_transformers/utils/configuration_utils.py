import inspect
import warnings
from collections import defaultdict
from inspect import Parameter, Signature
from typing import Any, TypeVar, Union

from qbruntime import Cluster, CoreId
from transformers.configuration_utils import PretrainedConfig

try:
    from transformers.configuration_utils import SpecificPretrainedConfigType
except ImportError:
    try:
        from transformers.configuration_utils import SpecificPreTrainedConfigType as SpecificPretrainedConfigType
    except ImportError:
        SpecificPretrainedConfigType = TypeVar(
            "SpecificPretrainedConfigType",
            bound=PretrainedConfig,
        )

from ...utils.core_mode import normalize_core_mode
from ...utils.npu_backend import MobilintNPUBackend

# NPU target-field normalization
# ------------------------------
# The canonical wire representation for NPU targets is fully-qualified:
#   - ``target_cores``:    list of ``"d:c:k"`` strings (device : cluster : core)
#   - ``target_clusters``: list of ``"d:c"`` strings   (device : cluster)
#
# ``dev_no`` is *syntactic sugar* for the device-prefix component. When both
# target lists are absent it is expanded into a canonical target list. When
# legacy 2-part items are provided, ``dev_no`` supplies the missing device
# prefix. Callers that need per-target device control should use the fully
# qualified form directly and pass ``dev_no`` as a list.


def _normalize_dev_list(dev_no: Any) -> list[int]:
    """Coerce a ``dev_no`` sugar value into a list of device indices."""
    if isinstance(dev_no, (list, tuple)):
        return [int(d) for d in dev_no]
    return [int(dev_no)]


def _migrate_target_cores(
    values: list, fallback_dev: int, dev_no_is_list: bool
) -> list[str]:
    """Migrate a mixed ``target_cores`` list to the canonical ``"d:c:k"`` form.

    Args:
        values: Raw ``target_cores`` list. Items may be ``CoreId`` objects,
            fully-qualified ``"d:c:k"`` strings, or legacy ``"c:k"`` strings.
        fallback_dev: Device prefix to apply to legacy items when ``dev_no``
            is a scalar.
        dev_no_is_list: ``True`` if the caller passed a list-shaped ``dev_no``.
            Legacy items are ambiguous in that case and must be rejected.

    Returns:
        A list of canonical ``"d:c:k"`` strings.

    Raises:
        ValueError: If entries mix legacy and new forms, or if a legacy entry
            appears with a list-shaped ``dev_no``.
        TypeError: If an entry is neither ``CoreId`` nor a string.
    """
    result: list[str] = []
    modes: set[str] = set()
    for v in values:
        if isinstance(v, CoreId):
            result.append(f"{fallback_dev}:{v.cluster.value}:{v.core.value}")
            modes.add("legacy")
            continue
        if not isinstance(v, str):
            raise TypeError(f"Unsupported target_cores entry: {v!r} ({type(v).__name__})")
        parts = v.split(":")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            result.append(v)
            modes.add("new")
        elif len(parts) == 2 and all(p.isdigit() for p in parts):
            if dev_no_is_list:
                raise ValueError(
                    f"Legacy target_cores item {v!r} is ambiguous when dev_no is a list; "
                    "use the fully-qualified 'd:c:k' form."
                )
            c_val, r_val = parts
            result.append(f"{fallback_dev}:{c_val}:{r_val}")
            modes.add("legacy")
        else:
            raise ValueError(f"Invalid target_cores entry: {v!r}")
    if len(modes) > 1:
        raise ValueError(
            "target_cores mixes legacy 'c:k' and canonical 'd:c:k' items; "
            "use one form for every entry."
        )
    return result


def _migrate_target_clusters(
    values: list, fallback_dev: int, dev_no_is_list: bool
) -> list[str]:
    """Migrate a mixed ``target_clusters`` list to the canonical ``"d:c"`` form.

    Args:
        values: Raw ``target_clusters`` list. Items may be ``Cluster`` objects,
            fully-qualified ``"d:c"`` strings, bare integers, or bare
            ``"c"`` strings (legacy).
        fallback_dev: Device prefix to apply to legacy items when ``dev_no``
            is a scalar.
        dev_no_is_list: ``True`` if the caller passed a list-shaped ``dev_no``.
            Legacy items are ambiguous in that case and must be rejected.

    Returns:
        A list of canonical ``"d:c"`` strings.

    Raises:
        ValueError: If entries mix legacy and new forms, or if a legacy entry
            appears with a list-shaped ``dev_no``.
        TypeError: If an entry is neither ``Cluster``, ``int``, nor a string.
    """
    result: list[str] = []
    modes: set[str] = set()
    for v in values:
        if isinstance(v, Cluster):
            result.append(f"{fallback_dev}:{v.value}")
            modes.add("legacy")
            continue
        if isinstance(v, bool):
            raise TypeError(f"Unsupported target_clusters entry: {v!r}")
        if isinstance(v, int):
            if dev_no_is_list:
                raise ValueError(
                    f"Legacy target_clusters int {v} is ambiguous when dev_no is a list; "
                    "use the fully-qualified 'd:c' form."
                )
            result.append(f"{fallback_dev}:{v}")
            modes.add("legacy")
            continue
        if not isinstance(v, str):
            raise TypeError(f"Unsupported target_clusters entry: {v!r} ({type(v).__name__})")
        parts = v.split(":")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            result.append(v)
            modes.add("new")
        elif len(parts) == 1 and parts[0].isdigit():
            if dev_no_is_list:
                raise ValueError(
                    f"Legacy target_clusters item {v!r} is ambiguous when dev_no is a list; "
                    "use the fully-qualified 'd:c' form."
                )
            result.append(f"{fallback_dev}:{v}")
            modes.add("legacy")
        else:
            raise ValueError(f"Invalid target_clusters entry: {v!r}")
    if len(modes) > 1:
        raise ValueError(
            "target_clusters mixes legacy and canonical items; use one form for every entry."
        )
    return result


def _expand_clusters_to_cores(clusters: list[str]) -> list[str]:
    """Expand each canonical ``"d:c"`` cluster string into its four ``"d:c:k"`` cores."""
    result: list[str] = []
    for cs in clusters:
        d_val, c_val = cs.split(":")
        for k in range(4):
            result.append(f"{d_val}:{c_val}:{k}")
    return result


def _fold_cores_to_clusters(cores: list[str], *, stacklevel: int) -> list[str]:
    """Fold canonical ``"d:c:k"`` core strings up to their unique ``"d:c"`` cluster prefixes.

    Emits a ``UserWarning`` for each cluster whose covered cores are a strict
    subset of ``{0, 1, 2, 3}``, so that callers know the effective target was
    rounded up.
    """
    per_cluster: dict[tuple[int, int], set[int]] = defaultdict(set)
    order: list[tuple[int, int]] = []
    for cs in cores:
        d_val, c_val, k_val = (int(x) for x in cs.split(":"))
        key = (d_val, c_val)
        if key not in per_cluster:
            order.append(key)
        per_cluster[key].add(k_val)
    for key in order:
        ks = per_cluster[key]
        if ks != {0, 1, 2, 3}:
            warnings.warn(
                f"target_cores {sorted(ks)} for cluster {key[0]}:{key[1]} do not cover all "
                f"four cores; rounded up to whole cluster.",
                UserWarning,
                stacklevel=stacklevel,
            )
    return [f"{d_val}:{c_val}" for (d_val, c_val) in order]


def _devices_from_targets(cores: list[str], clusters: list[str]) -> set[int]:
    """Return the set of device indices referenced by any canonical target string."""
    devs: set[int] = set()
    for cs in cores:
        devs.add(int(cs.split(":", 1)[0]))
    for cs in clusters:
        devs.add(int(cs.split(":", 1)[0]))
    return devs


def _validate_global8_coverage(clusters: list[str]) -> None:
    """Ensure every unique device in ``clusters`` covers both clusters 0 and 1."""
    per_dev: dict[int, set[int]] = defaultdict(set)
    for cs in clusters:
        d_val, c_val = (int(x) for x in cs.split(":"))
        per_dev[d_val].add(c_val)
    for d_val, cs_set in per_dev.items():
        if cs_set != {0, 1}:
            raise ValueError(
                f"core_mode='global8' requires both clusters for device {d_val}; got "
                f"clusters {sorted(cs_set)}."
            )


def _normalize_npu_target_kwargs(kwargs: dict[str, Any], prefix: str = "") -> None:
    """Normalize NPU target fields inside ``kwargs`` in place.

    Rewrites ``{prefix}target_cores`` and ``{prefix}target_clusters`` to the
    canonical fully-qualified representations, expands ``{prefix}dev_no`` sugar
    when both target lists are absent, and unifies grain to the field
    appropriate for ``{prefix}core_mode``. See task 8d97d for the full spec.

    Args:
        kwargs: The keyword-argument dict handed to a config mixin's
            ``__init__`` or ``__post_init__``. Mutated in place: the target
            fields are replaced with canonical values.
        prefix: Optional prefix that scopes the NPU keys (e.g. ``"encoder_"``,
            ``"vision_"``, ``"base_"``). Empty for the default backend.

    Raises:
        ValueError: For any inconsistency the spec calls out — mixed legacy
            and new items, list-shaped ``dev_no`` combined with legacy items,
            device-set mismatch, or incomplete global8 coverage.
        TypeError: When a target entry has an unsupported type.
    """
    core_mode_key = f"{prefix}core_mode"
    dev_no_key = f"{prefix}dev_no"
    cores_key = f"{prefix}target_cores"
    clusters_key = f"{prefix}target_clusters"

    core_mode = normalize_core_mode(kwargs.get(core_mode_key, "single"))
    dev_no = kwargs.get(dev_no_key, 0)
    dev_no_is_list = isinstance(dev_no, (list, tuple))
    dev_list = _normalize_dev_list(dev_no)
    fallback_dev = dev_list[0]
    dev_no_given = dev_no_key in kwargs

    raw_cores = kwargs.get(cores_key)
    raw_clusters = kwargs.get(clusters_key)

    cores = (
        _migrate_target_cores(list(raw_cores), fallback_dev, dev_no_is_list)
        if raw_cores
        else []
    )
    clusters = (
        _migrate_target_clusters(list(raw_clusters), fallback_dev, dev_no_is_list)
        if raw_clusters
        else []
    )

    if not cores and not clusters:
        # (2) dev_no sugar expansion when both target lists are absent.
        if core_mode == "single":
            cores = [
                f"{d}:{c}:{k}"
                for d in dev_list
                for c in (0, 1)
                for k in range(4)
            ]
        else:
            clusters = [f"{d}:{c}" for d in dev_list for c in (0, 1)]
    else:
        # (3) grain unification per core_mode.
        if core_mode == "single":
            if not cores and clusters:
                cores = _expand_clusters_to_cores(clusters)
        else:
            if not clusters and cores:
                clusters = _fold_cores_to_clusters(cores, stacklevel=4)

        # (5) device-set consistency check when the user explicitly set dev_no.
        if dev_no_given:
            target_devs = _devices_from_targets(cores, clusters)
            explicit_devs = set(dev_list)
            if target_devs != explicit_devs:
                raise ValueError(
                    f"target device set {sorted(target_devs)} does not match "
                    f"{dev_no_key} {sorted(explicit_devs)}."
                )

    # (4) global8 must cover both clusters on every unique device.
    if core_mode == "global8":
        _validate_global8_coverage(clusters)

    if cores:
        kwargs[cores_key] = cores
    if clusters:
        kwargs[clusters_key] = clusters


class MobilintConfigMixin(PretrainedConfig):
    # ``dev_no`` is exposed as syntactic sugar for the device-prefix component
    # of the canonical target strings. It accepts either a single device index
    # or a list of indices; downstream normalization expands it into
    # ``target_cores`` / ``target_clusters`` when the caller does not specify
    # targets directly.
    _NPU_SIGNATURE_FIELDS = (
        ("mxq_path", "", str),
        ("dev_no", 0, Union[int, list[int]]),
        ("max_batch_size", 1, int),
        ("core_mode", "single", str),
        ("target_cores", None, Any),
        ("target_clusters", None, Any),
        ("revision", None, Any),
        ("npu_prefill_chunk_size", None, Any),
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._augment_init_signature()

    @classmethod
    def _augment_init_signature(cls) -> None:
        """Expose Mobilint backend kwargs to upstream config introspection."""
        init = cls.__init__
        signature = inspect.signature(init)
        if any(name in signature.parameters for name, _, _ in cls._NPU_SIGNATURE_FIELDS):
            return

        parameters = list(signature.parameters.values())
        insert_at = next(
            (index for index, parameter in enumerate(parameters) if parameter.kind == Parameter.VAR_KEYWORD),
            len(parameters),
        )
        extra_parameters = [
            Parameter(name=name, kind=Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
            for name, default, annotation in cls._NPU_SIGNATURE_FIELDS
        ]
        init.__signature__ = Signature(parameters[:insert_at] + extra_parameters + parameters[insert_at:])

    def _ensure_npu_backend(self, kwargs: dict[str, Any]) -> None:
        if not hasattr(self, "npu_backend"):
            _normalize_npu_target_kwargs(kwargs, prefix="")
            self.npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="")

    def __init__(self, *args, **kwargs):
        self._ensure_npu_backend(kwargs)
        super().__init__(*args, **kwargs)

    def __post_init__(self, **kwargs: Any) -> None:
        self._ensure_npu_backend(kwargs)
        super().__post_init__(**kwargs)

    @property
    def mxq_path(self) -> str:
        return self.npu_backend.mxq_path

    @mxq_path.setter
    def mxq_path(self, value: str) -> None:
        self.npu_backend.mxq_path = value

    @property
    def dev_no(self) -> int:
        return self.npu_backend.dev_no

    @dev_no.setter
    def dev_no(self, value: int) -> None:
        self.npu_backend.dev_no = value

    @property
    def core_mode(self) -> str:
        return self.npu_backend.core_mode

    @core_mode.setter
    def core_mode(self, value: str) -> None:
        self.npu_backend.core_mode = value

    @property
    def max_batch_size(self) -> int:
        return self.npu_backend.max_batch_size

    @max_batch_size.setter
    def max_batch_size(self, value: int) -> None:
        self.npu_backend.max_batch_size = max(1, value)

    @property
    def target_cores(self) -> list:
        return self.npu_backend.target_cores

    @target_cores.setter
    def target_cores(self, values: list) -> None:
        self.npu_backend.target_cores = values

    @property
    def target_clusters(self) -> list:
        return self.npu_backend.target_clusters

    @target_clusters.setter
    def target_clusters(self, values: list) -> None:
        self.npu_backend.target_clusters = values

    @property
    def npu_prefill_chunk_size(self) -> Any:
        return self.__dict__.get("npu_prefill_chunk_size", None)

    @npu_prefill_chunk_size.setter
    def npu_prefill_chunk_size(self, value: Any) -> None:
        self.__dict__["npu_prefill_chunk_size"] = value

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        if hasattr(self, "npu_backend"):
            _ = d.pop("npu_backend", None)

        super()._remove_keys_not_serialized(d)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config and flatten Mobilint NPU backend fields into the top level.

        The ``npu_backend`` attribute is temporarily detached before delegating to the upstream
        :meth:`PretrainedConfig.to_dict` implementation so it is neither serialized as a nested
        object nor picked up by upstream diff/equality helpers. It is reattached in a ``finally``
        block, and its unprefixed field mapping is merged into the returned dictionary so callers
        see the individual NPU parameters (``mxq_path``, ``dev_no``, etc.) at the top level.

        Returns:
            A dictionary representation of the config with the Mobilint NPU backend fields
            merged in at the top level (no ``npu_backend`` key).
        """
        npu_backend = getattr(self, "npu_backend", None)
        if npu_backend is not None:
            del self.npu_backend
        try:
            output = super().to_dict()
        finally:
            if npu_backend is not None:
                self.npu_backend = npu_backend
        if npu_backend is not None:
            output.update(npu_backend.to_dict(prefix=""))
        return output


class MobilintEncoderDecoderConfigMixin(PretrainedConfig):
    def _ensure_encoder_decoder_npu_backends(self, kwargs: dict[str, Any]) -> None:
        if not hasattr(self, "encoder_npu_backend"):
            _normalize_npu_target_kwargs(kwargs, prefix="encoder_")
            self.encoder_npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="encoder_")

        if not hasattr(self, "decoder_npu_backend"):
            _normalize_npu_target_kwargs(kwargs, prefix="decoder_")
            self.decoder_npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="decoder_")

    def __init__(self, **kwargs):
        self._ensure_encoder_decoder_npu_backends(kwargs)
        super().__init__(**kwargs)

    def __post_init__(self, **kwargs: Any) -> None:
        self._ensure_encoder_decoder_npu_backends(kwargs)
        super().__post_init__(**kwargs)

    @property
    def encoder_mxq_path(self) -> str:
        return self.encoder_npu_backend.mxq_path

    @encoder_mxq_path.setter
    def encoder_mxq_path(self, value: str) -> None:
        self.encoder_npu_backend.mxq_path = value

    @property
    def encoder_dev_no(self) -> int:
        return self.encoder_npu_backend.dev_no

    @encoder_dev_no.setter
    def encoder_dev_no(self, value: int) -> None:
        self.encoder_npu_backend.dev_no = value

    @property
    def encoder_core_mode(self) -> str:
        return self.encoder_npu_backend.core_mode

    @encoder_core_mode.setter
    def encoder_core_mode(self, value: str) -> None:
        self.encoder_npu_backend.core_mode = value

    @property
    def encoder_max_batch_size(self) -> int:
        return self.encoder_npu_backend.max_batch_size

    @encoder_max_batch_size.setter
    def encoder_max_batch_size(self, value: int) -> None:
        self.encoder_npu_backend.max_batch_size = max(1, value)

    @property
    def encoder_target_cores(self) -> list:
        return self.encoder_npu_backend.target_cores

    @encoder_target_cores.setter
    def encoder_target_cores(self, values: list) -> None:
        self.encoder_npu_backend.target_cores = values

    @property
    def encoder_target_clusters(self) -> list:
        return self.encoder_npu_backend.target_clusters

    @encoder_target_clusters.setter
    def encoder_target_clusters(self, values: list) -> None:
        self.encoder_npu_backend.target_clusters = values

    @property
    def decoder_mxq_path(self) -> str:
        return self.decoder_npu_backend.mxq_path

    @decoder_mxq_path.setter
    def decoder_mxq_path(self, value: str) -> None:
        self.decoder_npu_backend.mxq_path = value

    @property
    def decoder_dev_no(self) -> int:
        return self.decoder_npu_backend.dev_no

    @decoder_dev_no.setter
    def decoder_dev_no(self, value: int) -> None:
        self.decoder_npu_backend.dev_no = value

    @property
    def decoder_core_mode(self) -> str:
        return self.decoder_npu_backend.core_mode

    @decoder_core_mode.setter
    def decoder_core_mode(self, value: str) -> None:
        self.decoder_npu_backend.core_mode = value

    @property
    def decoder_max_batch_size(self) -> int:
        return self.decoder_npu_backend.max_batch_size

    @decoder_max_batch_size.setter
    def decoder_max_batch_size(self, value: int) -> None:
        self.decoder_npu_backend.max_batch_size = max(1, value)

    @property
    def decoder_target_cores(self) -> list:
        return self.decoder_npu_backend.target_cores

    @decoder_target_cores.setter
    def decoder_target_cores(self, values: list) -> None:
        self.decoder_npu_backend.target_cores = values

    @property
    def decoder_target_clusters(self) -> list:
        return self.decoder_npu_backend.target_clusters

    @decoder_target_clusters.setter
    def decoder_target_clusters(self, values: list) -> None:
        self.decoder_npu_backend.target_clusters = values

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        if hasattr(self, "encoder_npu_backend"):
            _ = d.pop("encoder_npu_backend", None)

        if hasattr(self, "decoder_npu_backend"):
            _ = d.pop("decoder_npu_backend", None)

        super()._remove_keys_not_serialized(d)

    def to_dict(self):
        output = super().to_dict()

        if hasattr(self, "encoder_npu_backend"):
            output.update(self.encoder_npu_backend.to_dict(prefix="encoder_"))
        if hasattr(self, "decoder_npu_backend"):
            output.update(self.decoder_npu_backend.to_dict(prefix="decoder_"))

        return output

    def get_text_config(self, decoder=None, encoder=None) -> "PretrainedConfig":
        return self


class MobilintVisionTextConfigMixin(PretrainedConfig):
    sub_configs = {"vision_config": MobilintConfigMixin, "text_config": MobilintConfigMixin}

    _SUB_BACKEND_FIELDS = (
        "mxq_path",
        "dev_no",
        "max_batch_size",
        "core_mode",
        "target_cores",
        "target_clusters",
        "npu_prefill_chunk_size",
    )

    @classmethod
    def _split_sub_backend_kwargs(cls, kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pop ``text_*`` / ``vision_*`` NPU keys out of ``kwargs`` in place.

        Upstream composite configs (e.g. ``Qwen2VLConfig``, ``BlipConfig``) may
        call ``PretrainedConfig.__init__(**kwargs)`` before instantiating their
        sub-configs, which triggers the prefixed property setters on this mixin
        while ``self.text_config`` / ``self.vision_config`` do not yet exist.
        Removing the keys up front and re-applying them after the sub-configs
        are built avoids that ordering hazard.
        """
        text_kwargs: dict[str, Any] = {}
        vision_kwargs: dict[str, Any] = {}
        for field in cls._SUB_BACKEND_FIELDS:
            text_key = f"text_{field}"
            if text_key in kwargs:
                text_kwargs[field] = kwargs.pop(text_key)
            vision_key = f"vision_{field}"
            if vision_key in kwargs:
                vision_kwargs[field] = kwargs.pop(vision_key)
        return text_kwargs, vision_kwargs

    def _apply_sub_backend_kwargs(
        self, text_kwargs: dict[str, Any], vision_kwargs: dict[str, Any]
    ) -> None:
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            for key, value in text_kwargs.items():
                setattr(text_config, key, value)
        vision_config = getattr(self, "vision_config", None)
        if vision_config is not None:
            for key, value in vision_kwargs.items():
                setattr(vision_config, key, value)

    @PretrainedConfig.name_or_path.setter
    def name_or_path(self, value):
        PretrainedConfig.name_or_path.fset(self, value)
        vision_config = getattr(self, "vision_config", None)
        if vision_config is not None:
            vision_config.name_or_path = value
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            text_config.name_or_path = value

    @property
    def vision_mxq_path(self) -> str:
        return self.vision_config.mxq_path

    @vision_mxq_path.setter
    def vision_mxq_path(self, value: str) -> None:
        self.vision_config.mxq_path = value

    @property
    def vision_dev_no(self) -> int:
        return self.vision_config.dev_no

    @vision_dev_no.setter
    def vision_dev_no(self, value: int) -> None:
        self.vision_config.dev_no = value

    @property
    def vision_core_mode(self) -> str:
        return self.vision_config.core_mode

    @vision_core_mode.setter
    def vision_core_mode(self, value: str) -> None:
        self.vision_config.core_mode = value

    @property
    def vision_max_batch_size(self) -> int:
        return self.vision_config.max_batch_size

    @vision_max_batch_size.setter
    def vision_max_batch_size(self, value: int) -> None:
        self.vision_config.max_batch_size = max(1, value)

    @property
    def vision_target_cores(self) -> list:
        return self.vision_config.target_cores

    @vision_target_cores.setter
    def vision_target_cores(self, values: list) -> None:
        self.vision_config.target_cores = values

    @property
    def vision_target_clusters(self) -> list:
        return self.vision_config.target_clusters

    @vision_target_clusters.setter
    def vision_target_clusters(self, values: list) -> None:
        self.vision_config.target_clusters = values

    @property
    def text_mxq_path(self) -> str:
        return self.text_config.mxq_path

    @text_mxq_path.setter
    def text_mxq_path(self, value: str) -> None:
        self.text_config.mxq_path = value

    @property
    def text_dev_no(self) -> int:
        return self.text_config.dev_no

    @text_dev_no.setter
    def text_dev_no(self, value: int) -> None:
        self.text_config.dev_no = value

    @property
    def text_core_mode(self) -> str:
        return self.text_config.core_mode

    @text_core_mode.setter
    def text_core_mode(self, value: str) -> None:
        self.text_config.core_mode = value

    @property
    def text_max_batch_size(self) -> int:
        return self.text_config.max_batch_size

    @text_max_batch_size.setter
    def text_max_batch_size(self, value: int) -> None:
        self.text_config.max_batch_size = max(1, value)

    @property
    def text_target_cores(self) -> list:
        return self.text_config.target_cores

    @text_target_cores.setter
    def text_target_cores(self, values: list) -> None:
        self.text_config.target_cores = values

    @property
    def text_target_clusters(self) -> list:
        return self.text_config.target_clusters

    @text_target_clusters.setter
    def text_target_clusters(self, values: list) -> None:
        self.text_config.target_clusters = values

    @property
    def text_npu_prefill_chunk_size(self) -> Any:
        return self.text_config.npu_prefill_chunk_size

    @text_npu_prefill_chunk_size.setter
    def text_npu_prefill_chunk_size(self, value: Any) -> None:
        self.text_config.npu_prefill_chunk_size = value

    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> Union["MobilintVisionTextConfigMixin", tuple["MobilintVisionTextConfigMixin", dict[str, Any]]]:
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config: MobilintVisionTextConfigMixin
        unused_kwargs: dict[str, Any]
        config, unused_kwargs = super().from_dict(config_dict, return_unused_kwargs=True, **kwargs)  # type: ignore

        for sub_config in (config.text_config, config.vision_config):
            sub_config.name_or_path = config.name_or_path

            revision = getattr(config, "revision", None)
            if revision:
                sub_config.revision = revision

            commit_hash = getattr(config, "_commit_hash", None)
            if commit_hash:
                sub_config._commit_hash = commit_hash

        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: MobilintConfigMixin,
        vision_config: MobilintConfigMixin,
        **kwargs,
    ):
        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )


class MobilintEagle3ConfigMixin(PretrainedConfig):
    """Config mixin for EAGLE-3 models with base/draft/fc backends."""

    sub_configs = {"draft_config": MobilintConfigMixin}
    _EAGLE3_BACKEND_FIELDS = (
        "mxq_path",
        "dev_no",
        "max_batch_size",
        "core_mode",
        "target_cores",
        "target_clusters",
        "revision",
        "commit_hash",
    )
    _EAGLE3_RUNTIME_FIELDS = (
        ("eagle3_tree_depth", 5, int),
        ("eagle3_tree_top_k", 8, int),
        ("eagle3_npu_chunk_size", 192, int),
    )

    @classmethod
    def _get_draft_config_class(cls) -> type[PretrainedConfig]:
        return MobilintConfigMixin

    def _init_or_coerce_draft_config(self, draft_config: Any | None) -> None:
        draft_config_cls = self._get_draft_config_class()

        if draft_config is None:
            coerced_draft_config = draft_config_cls()
        elif isinstance(draft_config, dict):
            coerced_draft_config = draft_config_cls(**draft_config)
        elif isinstance(draft_config, draft_config_cls):
            coerced_draft_config = draft_config
        else:
            raise TypeError(
                f"draft_config must be None, dict, or {draft_config_cls.__name__}; got {type(draft_config).__name__}"
            )

        self.draft_config = coerced_draft_config
        self.draft_config.name_or_path = self.name_or_path

    def _ensure_eagle3_npu_backends(self, kwargs: dict[str, Any]) -> None:
        def _resolve_backend_kwargs(prefix: str) -> dict[str, Any]:
            backend_kwargs: dict[str, Any] = {}
            for field_name in self._EAGLE3_BACKEND_FIELDS:
                prefixed_key = f"{prefix}{field_name}"
                if prefixed_key in kwargs:
                    backend_kwargs[prefixed_key] = kwargs[prefixed_key]
                    continue

                if field_name in kwargs:
                    backend_kwargs[prefixed_key] = kwargs[field_name]
            return backend_kwargs

        if not hasattr(self, "base_npu_backend"):
            base_kwargs = _resolve_backend_kwargs("base_")
            _normalize_npu_target_kwargs(base_kwargs, prefix="base_")
            self.base_npu_backend = MobilintNPUBackend.from_dict(base_kwargs, prefix="base_")
        if not hasattr(self, "draft_npu_backend"):
            draft_kwargs = _resolve_backend_kwargs("draft_")
            _normalize_npu_target_kwargs(draft_kwargs, prefix="draft_")
            self.draft_npu_backend = MobilintNPUBackend.from_dict(draft_kwargs, prefix="draft_")
        if not hasattr(self, "fc_npu_backend"):
            fc_kwargs = _resolve_backend_kwargs("fc_")
            _normalize_npu_target_kwargs(fc_kwargs, prefix="fc_")
            self.fc_npu_backend = MobilintNPUBackend.from_dict(fc_kwargs, prefix="fc_")

    def _ensure_eagle3_runtime_fields(self, kwargs: dict[str, Any]) -> None:
        for field_name, default_value, _annotation in self._EAGLE3_RUNTIME_FIELDS:
            if hasattr(self, field_name):
                continue
            value = kwargs.pop(field_name, default_value)
            self.__dict__[field_name] = self._coerce_positive_runtime_int(
                field_name,
                value,
                default_value=default_value,
            )

    @staticmethod
    def _coerce_positive_runtime_int(field_name: str, value: Any, *, default_value: int) -> int:
        """Coerce runtime integer fields and reject invalid ranges."""
        candidate = default_value if value is None else value
        try:
            coerced = int(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer, got {candidate!r}") from exc
        if coerced <= 0:
            raise ValueError(f"{field_name} must be > 0, got {coerced}")
        return coerced

    def __init__(self, **kwargs):
        draft_config = kwargs.pop("draft_config", None)
        self._ensure_eagle3_npu_backends(kwargs)
        self._ensure_eagle3_runtime_fields(kwargs)
        super().__init__(**kwargs)
        self._init_or_coerce_draft_config(draft_config)

    def __post_init__(self, **kwargs: Any) -> None:
        draft_config = kwargs.pop("draft_config", getattr(self, "draft_config", None))
        self._ensure_eagle3_npu_backends(kwargs)
        self._ensure_eagle3_runtime_fields(kwargs)
        super().__post_init__(**kwargs)
        self._init_or_coerce_draft_config(draft_config)

    @PretrainedConfig.name_or_path.setter
    def name_or_path(self, value: str):
        PretrainedConfig.name_or_path.fset(self, value)
        draft_config = getattr(self, "draft_config", None)
        if draft_config is not None:
            draft_config.name_or_path = value

    @property
    def base_dev_no(self) -> int:
        return self.base_npu_backend.dev_no

    @base_dev_no.setter
    def base_dev_no(self, value: int) -> None:
        self.base_npu_backend.dev_no = value

    @property
    def draft_dev_no(self) -> int:
        return self.draft_npu_backend.dev_no

    @draft_dev_no.setter
    def draft_dev_no(self, value: int) -> None:
        self.draft_npu_backend.dev_no = value

    @property
    def fc_dev_no(self) -> int:
        return self.fc_npu_backend.dev_no

    @fc_dev_no.setter
    def fc_dev_no(self, value: int) -> None:
        self.fc_npu_backend.dev_no = value

    @property
    def base_max_batch_size(self) -> int:
        return self.base_npu_backend.max_batch_size

    @base_max_batch_size.setter
    def base_max_batch_size(self, value: int) -> None:
        self.base_npu_backend.max_batch_size = max(1, value)

    @property
    def draft_max_batch_size(self) -> int:
        return self.draft_npu_backend.max_batch_size

    @draft_max_batch_size.setter
    def draft_max_batch_size(self, value: int) -> None:
        self.draft_npu_backend.max_batch_size = max(1, value)

    @property
    def fc_max_batch_size(self) -> int:
        return self.fc_npu_backend.max_batch_size

    @fc_max_batch_size.setter
    def fc_max_batch_size(self, value: int) -> None:
        self.fc_npu_backend.max_batch_size = max(1, value)

    @property
    def base_core_mode(self) -> str:
        return self.base_npu_backend.core_mode

    @base_core_mode.setter
    def base_core_mode(self, value: str) -> None:
        self.base_npu_backend.core_mode = value

    @property
    def draft_core_mode(self) -> str:
        return self.draft_npu_backend.core_mode

    @draft_core_mode.setter
    def draft_core_mode(self, value: str) -> None:
        self.draft_npu_backend.core_mode = value

    @property
    def fc_core_mode(self) -> str:
        return self.fc_npu_backend.core_mode

    @fc_core_mode.setter
    def fc_core_mode(self, value: str) -> None:
        self.fc_npu_backend.core_mode = value

    @property
    def base_target_cores(self) -> list[str]:
        return self.base_npu_backend.target_cores

    @base_target_cores.setter
    def base_target_cores(self, values: list[str]) -> None:
        self.base_npu_backend.target_cores = values

    @property
    def draft_target_cores(self) -> list[str]:
        return self.draft_npu_backend.target_cores

    @draft_target_cores.setter
    def draft_target_cores(self, values: list[str]) -> None:
        self.draft_npu_backend.target_cores = values

    @property
    def fc_target_cores(self) -> list[str]:
        return self.fc_npu_backend.target_cores

    @fc_target_cores.setter
    def fc_target_cores(self, values: list[str]) -> None:
        self.fc_npu_backend.target_cores = values

    @property
    def base_target_clusters(self) -> list[int]:
        return self.base_npu_backend.target_clusters

    @base_target_clusters.setter
    def base_target_clusters(self, values: list[int]) -> None:
        self.base_npu_backend.target_clusters = values

    @property
    def draft_target_clusters(self) -> list[int]:
        return self.draft_npu_backend.target_clusters

    @draft_target_clusters.setter
    def draft_target_clusters(self, values: list[int]) -> None:
        self.draft_npu_backend.target_clusters = values

    @property
    def fc_target_clusters(self) -> list[int]:
        return self.fc_npu_backend.target_clusters

    @fc_target_clusters.setter
    def fc_target_clusters(self, values: list[int]) -> None:
        self.fc_npu_backend.target_clusters = values

    @property
    def base_mxq_path(self) -> str:
        return self.base_npu_backend.mxq_path

    @base_mxq_path.setter
    def base_mxq_path(self, value: str) -> None:
        self.base_npu_backend.mxq_path = value

    @property
    def draft_mxq_path(self) -> str:
        return self.draft_npu_backend.mxq_path

    @draft_mxq_path.setter
    def draft_mxq_path(self, value: str) -> None:
        self.draft_npu_backend.mxq_path = value

    @property
    def fc_mxq_path(self) -> str:
        return self.fc_npu_backend.mxq_path

    @fc_mxq_path.setter
    def fc_mxq_path(self, value: str) -> None:
        self.fc_npu_backend.mxq_path = value

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        _ = d.pop("base_npu_backend", None)
        _ = d.pop("draft_npu_backend", None)
        _ = d.pop("fc_npu_backend", None)
        super()._remove_keys_not_serialized(d)

    def to_dict(self):
        output = super().to_dict()
        output.update(self.base_npu_backend.to_dict(prefix="base_"))
        output.update(self.draft_npu_backend.to_dict(prefix="draft_"))
        output.update(self.fc_npu_backend.to_dict(prefix="fc_"))
        return output
