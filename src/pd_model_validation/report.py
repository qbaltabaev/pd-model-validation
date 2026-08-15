"""Validation report container and lightweight exports."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .types import ValidationResult


def _json_safe(value: Any) -> Any:
    """Recursively convert values to strict, interoperable JSON types."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return str(value)


@dataclass
class ValidationReport:
    """Results plus supporting diagnostic tables."""

    results: list[ValidationResult]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return all validation findings as a tidy data frame."""
        return pd.DataFrame([result.to_dict() for result in self.results])

    def summary(self) -> pd.Series:
        """Count findings by traffic-light status."""
        frame = self.to_frame()
        if frame.empty:
            return pd.Series(dtype=int)
        return frame["status"].value_counts()

    def has_failures(self, *, include_amber: bool = False) -> bool:
        """Return whether the report contains RED, or optionally AMBER, findings."""
        failure_statuses = {"RED", "AMBER"} if include_amber else {"RED"}
        return any(result.status.value in failure_statuses for result in self.results)

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize results and supporting tables, optionally writing a file."""
        payload = {
            "metadata": self.metadata,
            "results": [result.to_dict() for result in self.results],
            "tables": {
                name: table.to_dict(orient="records") for name, table in self.tables.items()
            },
        }
        text = json.dumps(_json_safe(payload), indent=2, allow_nan=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def to_csv(self, path: str | Path) -> None:
        """Write the tidy validation findings to CSV."""
        self.to_frame().to_csv(path, index=False)

    def to_html(
        self, path: str | Path | None = None, *, title: str = "PD validation report"
    ) -> str:
        """Render a self-contained HTML report, optionally writing a file."""
        sections = [
            f"<h1>{html.escape(title)}</h1>",
            "<h2>Status summary</h2>",
            self.summary()
            .rename_axis("status")
            .reset_index(name="count")
            .to_html(index=False, escape=True),
            "<h2>Run metadata</h2>",
            f"<pre>{html.escape(json.dumps(_json_safe(self.metadata), indent=2))}</pre>",
            "<h2>Validation results</h2>",
            self.to_frame().to_html(index=False, escape=True),
        ]
        for name, table in self.tables.items():
            sections.extend(
                [
                    f"<h2>{html.escape(name.replace('_', ' ').title())}</h2>",
                    table.to_html(index=False, escape=True),
                ]
            )
        document = (
            "<!doctype html><html><head><meta charset='utf-8'><title>"
            + html.escape(title)
            + "</title><style>body{font:14px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#17202a}table{border-collapse:collapse;width:100%;margin-bottom:32px}th,td{border:1px solid #d9dee3;padding:6px;text-align:right}th{background:#f3f4f6}pre{background:#f6f8fa;padding:16px;overflow:auto}h1,h2{text-align:left}</style></head><body>"
            + "".join(sections)
            + "</body></html>"
        )
        if path is not None:
            Path(path).write_text(document, encoding="utf-8")
        return document
