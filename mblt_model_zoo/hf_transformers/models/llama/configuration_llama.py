from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintLlamaConfig(MobilintConfigMixin, LlamaConfig):
    model_type = "mobilint-llama"

    @staticmethod
    def _get_strict_compatible_hidden_size(hidden_size: object, num_attention_heads: object) -> int | None:
        """Return a temporary hidden size that satisfies upstream strict validation."""
        if not isinstance(hidden_size, int) or not isinstance(num_attention_heads, int):
            return None
        if num_attention_heads <= 0 or hidden_size % num_attention_heads == 0:
            return None
        return num_attention_heads * max(1, hidden_size // num_attention_heads)

    def validate_architecture(self) -> None:
        """Validate Mobilint Llama architecture constraints.

        Mobilint Llama artifacts may use non-square query projections, so
        ``hidden_size`` is not required to be divisible by ``num_attention_heads``.
        This overrides the upstream Llama validator while keeping lightweight
        validation for explicitly configured attention head dimensions.
        """
        head_dim = getattr(self, "head_dim", None)
        if head_dim is not None and head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}.")

    def __init__(self, **kwargs,):
        """Initialize the config while allowing non-square Mobilint query projections."""
        actual_hidden_size = kwargs.get("hidden_size")
        temporary_hidden_size = self._get_strict_compatible_hidden_size(
            actual_hidden_size,
            kwargs.get("num_attention_heads"),
        )

        if temporary_hidden_size is not None:
            kwargs = dict(kwargs)
            kwargs["hidden_size"] = temporary_hidden_size

        super().__init__(**kwargs)

        if temporary_hidden_size is not None:
            self.hidden_size = actual_hidden_size
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-llama", MobilintLlamaConfig)
