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

    def classify(self, value: float) -> Status:
        if not np.isfinite(value):
            return Status.NOT_APPLICABLE
        if self.direction == "higher":
            if self.green < self.amber:
                raise ValueError("green must be >= amber when higher values are better")
            if value >= self.green:
                return Status.GREEN
            if value >= self.amber:
                return Status.AMBER
            return Status.RED
        if self.green > self.amber:
            raise ValueError("green must be <= amber when lower values are better")
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
    segment: str | None = None
    period: str | None = None
    n_obs: int | None = None
    threshold_green: float | None = None
    threshold_amber: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data
