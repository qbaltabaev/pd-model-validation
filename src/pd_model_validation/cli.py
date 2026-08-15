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
    parser.add_argument("--group", help="optional rating-grade, pool, or segment column")
    parser.add_argument("--output", type=Path, default=Path("pd-validation-report.html"))
    parser.add_argument(
        "--format",
        choices=["html", "json", "csv"],
        help="output format; inferred from --output when omitted",
    )
    parser.add_argument(
        "--fail-on",
        choices=["never", "red", "amber"],
        default="red",
        help="return exit code 2 for validation findings at this severity",
    )
    parser.add_argument("--model-id", help="model identifier recorded in report metadata")
    parser.add_argument("--model-version", help="model version recorded in report metadata")
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
        group=args.group,
        metadata={"model_id": args.model_id, "model_version": args.model_version},
    )
    output_format = args.format or args.output.suffix.lstrip(".").lower() or "html"
    if output_format == "html":
        report.to_html(args.output)
    elif output_format == "json":
        report.to_json(args.output)
    elif output_format == "csv":
        report.to_csv(args.output)
    else:
        raise ValueError("output extension must be .html, .json, or .csv, or pass --format")
    print(report.summary().to_string())
    print(f"Report written to {args.output}")
    if args.fail_on == "red" and report.has_failures():
        return 2
    if args.fail_on == "amber" and report.has_failures(include_amber=True):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
