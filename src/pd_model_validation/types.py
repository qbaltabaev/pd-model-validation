"""Shared result types and traffic-light thresholds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np


class Status(str, Enum):
    """Traffic-light status for a validation result."""

    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"
    INFO = "INFO"
    NOT_APPLICABLE = "N/A"


Direction = Literal["higher", "lower"]


@dataclass(frozen=True)
class Thresholds:
    """Two cut-offs used to map a metric to a traffic-light status.

    For ``direction='higher'``, values at or above ``green`` pass and
    values below ``amber`` fail. The inequalities are reversed when lower
    values are better.
    """

    green: float
    amber: float
    direction: Direction = "higher"

    def __post_init__(self) -> None:
        """Reject ambiguous or internally inconsistent thresholds."""
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'")
        if not np.isfinite(self.green) or not np.isfinite(self.amber):
            raise ValueError("green and amber thresholds must be finite")
        if self.direction == "higher" and self.green < self.amber:
            raise ValueError("green must be >= amber when higher values are better")
        if self.direction == "lower" and self.green > self.amber:
            raise ValueError("green must be <= amber when lower values are better")

    def classify(self, value: float) -> Status:
        """Map a numeric metric value to GREEN, AMBER, RED, or N/A."""
        if not np.isfinite(value):
            return Status.NOT_APPLICABLE
        if self.direction == "higher":
            if value >= self.green:
                return Status.GREEN
            if value >= self.amber:
                return Status.AMBER
            return Status.RED
        if value <= self.green:
            return Status.GREEN
        if value <= self.amber:
            return Status.AMBER
        return Status.RED


@dataclass(frozen=True)
class ValidationResult:
    """One machine-readable validation finding."""

    test: str
    value: float
    status: Status = Status.INFO
    scope: str = "model"
    feature: str | None = None
    group: str | None = None
    segment: str | None = None
    period: str | None = None
    n_obs: int | None = None
    threshold_green: float | None = None
    threshold_amber: float | None = None
    p_value: float | None = None
    confidence_lower: float | None = None
    confidence_upper: float | None = None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        data = asdict(self)
        data["status"] = self.status.value
        return data
