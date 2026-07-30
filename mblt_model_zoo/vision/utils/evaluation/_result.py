"""Shared structural contract for Vision evaluator results."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EvaluationResult(Protocol):
    """Result object exposing the primary and secondary benchmark scores."""

    @property
    def primary_score(self) -> float:
        """Return the evaluator's primary score."""

        ...

    @property
    def secondary_score(self) -> float:
        """Return the evaluator's secondary score."""

        ...
