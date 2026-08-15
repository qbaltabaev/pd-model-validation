"""Validation report container and lightweight exports."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .types import ValidationResult


@dataclass
class ValidationReport:
    """Results plus supporting diagnostic tables."""

    results: list[ValidationResult]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([result.to_dict() for result in self.results])

    def summary(self) -> pd.Series:
        return self.to_frame()["status"].value_counts()

    def to_json(self, path: str | Path | None = None) -> str:
        payload = {
            "results": [result.to_dict() for result in self.results],
            "tables": {
                name: table.to_dict(orient="records") for name, table in self.tables.items()
            },
        }
        text = json.dumps(payload, indent=2, default=str)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def to_csv(self, path: str | Path) -> None:
        self.to_frame().to_csv(path, index=False)

    def to_html(
        self, path: str | Path | None = None, *, title: str = "PD validation report"
    ) -> str:
        sections = [
            f"<h1>{html.escape(title)}</h1>",
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
            + "</title><style>body{font:14px system-ui;max-width:1200px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;width:100%;margin-bottom:32px}th,td{border:1px solid #ddd;padding:6px;text-align:right}th{background:#f3f4f6}</style></head><body>"
            + "".join(sections)
            + "</body></html>"
        )
        if path is not None:
            Path(path).write_text(document, encoding="utf-8")
        return document
