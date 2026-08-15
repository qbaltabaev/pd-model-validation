# Contributing

1. Open an issue for material methodological or API changes.
2. Create a focused branch and add tests for changed behavior.
3. Run `ruff check .`, `ruff format --check .`, and `pytest --cov`.
4. Update documentation when definitions, thresholds, or public APIs change.

Use synthetic or public data only. Keep functions deterministic where practical, document statistical assumptions, and prefer explicit errors over silent corrections.
