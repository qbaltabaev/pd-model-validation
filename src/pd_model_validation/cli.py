"""Command-line interface for validating CSV datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .validator import PDValidator


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Validate a probability-of-default model")
    parser.add_argument("reference", type=Path, help="reference/development CSV")
    parser.add_argument("--current", type=Path, help="current/OOT CSV")
    parser.add_argument("--target", required=True, help="binary default target column")
    parser.add_argument("--probability", required=True, help="predicted PD column")
    parser.add_argument("--features", nargs="*", default=[], help="model feature columns")
    parser.add_argument("--date", help="observation date column")
    parser.add_argument("--output", type=Path, default=Path("pd-validation-report.html"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run CSV-based validation and write an HTML report."""
    args = build_parser().parse_args(argv)
    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current) if args.current else None
    report = PDValidator().validate(
        reference,
        target=args.target,
        probability=args.probability,
        current=current,
        features=args.features,
        date=args.date,
    )
    report.to_html(args.output)
    print(report.summary().to_string())
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
