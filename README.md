# PD Model Validation

[![CI](https://github.com/qbaltabaev/pd-model-validation/actions/workflows/ci.yml/badge.svg)](https://github.com/qbaltabaev/pd-model-validation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A compact, transparent Python toolkit providing statistical diagnostics for probability-of-default (PD) development, validation, and monitoring. It supports outcome backtesting, feature diagnostics, temporal analysis, unlabeled population monitoring, and Weight of Evidence (WoE) transformation without imposing organization-specific column names.

> This library provides statistical tools, not regulatory approval. Thresholds are deliberately configurable and should be aligned with the portfolio, default definition, rating philosophy, materiality, and applicable governance.

## What is included

| Area | Methods |
|---|---|
| Discrimination | ROC AUC, normalized Gini, KS, stratified bootstrap Gini interval |
| Calibration | O/E with confidence interval, calibration intercept/slope with fit diagnostics, Brier score, log loss, tie-preserving calibration tables, grouped grade/pool backtesting, Hosmer–Lemeshow |
| Stability | Numeric and categorical PSI, bin-level PSI contributions, feature and score stability, relative Gini degradation |
| Time validation | Monthly/quarterly volumes, default rates, average PD, and Gini |
| Features | WoE transformation, information value, univariate Gini, Spearman/Pearson correlation, VIF, logistic coefficient Wald tests |
| Reporting | Tidy findings, bootstrap intervals, run provenance and data fingerprints, strict JSON, HTML/CSV, automation-aware CLI |

## Installation

Use Python 3.14 for new environments. Until a tagged package release is available,
pin an immutable Git commit rather than installing a moving branch:

```bash
uv add "pd-model-validation @ git+https://github.com/qbaltabaev/pd-model-validation.git@<commit-sha>"
```

For local development, install [uv](https://docs.astral.sh/uv/) and run:

```bash
git clone https://github.com/qbaltabaev/pd-model-validation.git
cd pd-model-validation
uv sync --dev
uv run pytest
```

## Quick start

```python
import pandas as pd
from pd_model_validation import PDValidator

validation = pd.read_csv("validation.csv")
recent_applications = pd.read_csv("recent_applications.csv")

report = PDValidator().validate(
    validation,
    current=recent_applications,  # target may be absent while outcomes mature
    target="pd90_12m",
    probability="predicted_pd",
    features=["utilization", "age_months", "arrears_count"],
    date="application_date",
)

print(report.to_frame())
print(report.tables["feature_stability"])
report.to_html("pd-validation-report.html")
```

Current data does not need a mature target for PSI monitoring. Use
`PDValidator.monitor_drift(...)` when neither input contains outcomes. Each result
records the metric, status, sample, scope, optional feature/group/period,
observation count, thresholds, p-value or confidence interval, reason, and
details. Report metadata captures configuration, caller context, sample sizes,
package version, run time, and SHA-256 data fingerprints.

### WoE transformation

```python
from pd_model_validation import WoETransformer

transformer = WoETransformer(bins=10, handle_unknown="zero")
X_train_woe = transformer.fit_transform(X_train, y_train)
X_oot_woe = transformer.transform(X_oot)
print(transformer.iv_)
```

Breakpoints and mappings are fitted only on the reference sample. Missing values receive a dedicated bin. Unseen categories can receive neutral WoE or raise an error through `handle_unknown`; their population movement remains visible in PSI contribution tables.

### Front-book underwriting dataset and model

The repository includes [160 synthetic credit-card applications](examples/data/underwriting_pd_data.csv). The example uses 90 early applications for development, 30 later applications for holdout validation, and 40 applications for out-of-time assessment. The `pd90_12m` outcome identifies applications that reached at least 90 days past due within 12 months. Features cover income, employment, requested limit, card exposure and utilization, mortgage and auto payments, total loan and deposit balances, recent delinquencies, credit history, indebtedness, inquiries, and bankruptcy. See the [data dictionary and modeling frame](docs/example-data.md) for precise definitions.

The data represents a front-book underwriting use case and contains no real customers, organizations, locations, currencies, or source-system identifiers. Regenerate it deterministically or run the fitted model and complete validation with:

```bash
uv run python examples/generate_underwriting_data.py
uv run python examples/underwriting_model.py
```

The model fits preprocessing and logistic regression only on development data, then scores the distinct holdout and out-of-time samples. It produces `underwriting-pd-validation.html`; its functions can also be imported in tests or notebooks.

### Command line

```bash
pdvalidate development.csv \
  --current out_of_time.csv \
  --target pd90_12m \
  --probability predicted_pd \
  --features utilization age_months arrears_count \
  --date application_date \
  --group rating_grade \
  --model-id card_application_pd \
  --model-version 2.1 \
  --fail-on red \
  --output validation.html
```

The CLI infers HTML, JSON, or CSV from the output extension and exits with code
`2` when the configured gate is breached. Pass `--fail-on never` for reporting-only runs.

## Thresholds

Defaults are pragmatic starting points, not universal policy. Override them explicitly:

```python
from pd_model_validation import PDValidator, Thresholds, ValidationConfig

config = ValidationConfig(
    gini=Thresholds(green=0.45, amber=0.30, direction="higher"),
    psi=Thresholds(green=0.10, amber=0.20, direction="lower"),
    threshold_profile="front-book-policy-2026",
)
validator = PDValidator(config)
```

See [methodology](docs/methodology.md) for definitions and interpretation, and [validation workflow](docs/validation-workflow.md) for a governance-oriented checklist. A fully reproducible synthetic example is in [examples/quickstart.py](examples/quickstart.py).

## Repository map

- `src/pd_model_validation/`: typed, composable library code
- `tests/`: unit, edge-case, reporting, CLI, and executable-example tests
- `examples/`: deterministic data generator and leakage-safe model workflow
- `docs/methodology.md`: definitions, assumptions, and interpretation limits
- `docs/validation-workflow.md`: professional validation evidence checklist
- `docs/example-data.md`: synthetic underwriting data dictionary

## Design principles

- Reference-sample transformations are reused unchanged on validation samples.
- Functions reject invalid targets and probabilities instead of silently clipping them.
- Invalid targets or probabilities fail before metrics run; excluded incomplete pairs and effective observations are reported.
- Tied predicted probabilities remain in the same calibration bin.
- Current outcomes are optional for score and feature monitoring.
- Metrics are small, composable functions; the orchestrator is optional.
- Results are data frames and plain Python objects suitable for notebooks, pipelines, and audit evidence.
- The checked-in `uv.lock` and `.python-version` make development reproducible on Python 3.14 while CI tests Python 3.10–3.14 plus macOS and Windows smoke environments.

## License

MIT. See [LICENSE](LICENSE).
