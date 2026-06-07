"""Shared dataset helpers for benchmark scripts.

This module keeps optional dataset/audio dependencies lazily imported so benchmark
utilities that do not need dataset loading can still run without them.
"""

from __future__ import annotations

import io
import itertools
import math
from collections.abc import Iterable, Iterator
from typing import Any, Mapping


def resample_audio(audio_array: Any, source_rate: int, target_rate: int = 16000) -> Any:
    """Resample mono audio with a polyphase filter.

    Args:
        audio_array: Mono float audio samples.
        source_rate: Original sampling rate.
        target_rate: Desired sampling rate.

    Returns:
        Resampled mono audio array.
    """

    import numpy as np
    from scipy.signal import resample_poly

    if int(source_rate) == int(target_rate):
        return np.asarray(audio_array, dtype=np.float32)

    divisor = math.gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    resampled = resample_poly(np.asarray(audio_array, dtype=np.float32), up, down)
    return np.asarray(resampled, dtype=np.float32)


def load_streaming_audio_text_samples(
    *,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    audio_column: str = "audio",
    text_column: str = "text",
    id_column: str = "id",
    num_samples: int | None = None,
    seed: int = 0,
    target_sampling_rate: int = 16000,
) -> Iterable[dict[str, Any]]:
    """Load streaming dataset rows into benchmark-ready audio/text samples.

    Args:
        dataset_name: Hugging Face dataset name.
        dataset_config: Hugging Face dataset config name.
        dataset_split: Dataset split to iterate.
        audio_column: Column containing audio payload metadata.
        text_column: Column containing reference text.
        id_column: Column containing sample id.
        num_samples: Maximum number of rows to consume. ``None`` consumes the full split.
        seed: Shuffle seed for streaming datasets that support shuffling.
        target_sampling_rate: Desired output audio sampling rate.

    Returns:
        A streaming iterable of benchmark sample dictionaries with ``id``, ``audio``, and
        ``reference`` keys. When ``num_samples`` is set, the loader shuffles the streaming
        dataset when supported and then consumes only the requested prefix of decoded rows.

    Raises:
        ValueError: If a dataset row does not contain a readable audio payload.
    """

    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    def _decode_audio(raw_audio: Mapping[str, Any]) -> tuple[Any, int]:
        path = raw_audio.get("path")
        audio_bytes = raw_audio.get("bytes")
        if isinstance(audio_bytes, (bytes, bytearray)):
            audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        elif isinstance(path, str) and path:
            audio_array, sampling_rate = sf.read(path, dtype="float32")
        else:
            raise ValueError("Dataset audio row does not contain readable path or bytes payload.")

        if getattr(audio_array, "ndim", 1) > 1:
            audio_array = np.mean(audio_array, axis=1)

        sampling_rate = int(sampling_rate)
        if sampling_rate != int(target_sampling_rate):
            audio_array = resample_audio(audio_array, sampling_rate, int(target_sampling_rate))
            sampling_rate = int(target_sampling_rate)
        return audio_array, sampling_rate

    def _iter_decoded_rows(rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
        for index, row in enumerate(rows):
            audio_array, sampling_rate = _decode_audio(row[audio_column])
            yield {
                "id": str(row.get(id_column, index)),
                "audio": {"array": audio_array, "sampling_rate": sampling_rate},
                "reference": str(row.get(text_column, "")),
            }

    if dataset_config is None:
        dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True)
    if hasattr(dataset, "cast_column"):
        dataset = dataset.cast_column(audio_column, Audio(decode=False))
    if num_samples is None:
        return _iter_decoded_rows(dataset)

    sample_count = max(int(num_samples), 0)
    if sample_count == 0:
        return []

    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)
    return list(itertools.islice(_iter_decoded_rows(dataset), sample_count))