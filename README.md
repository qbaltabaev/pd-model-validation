# PD Model Validation

[![CI](https://github.com/qbaltabaev/pd-model-validation/actions/workflows/ci.yml/badge.svg)](https://github.com/qbaltabaev/pd-model-validation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A compact, transparent Python toolkit for developing, independently validating, and monitoring probability-of-default (PD) models. It supports model-level backtesting, feature diagnostics, temporal analysis, sample drift, and Weight of Evidence (WoE) transformation without imposing organization-specific column names or workflows.

> This library provides statistical tools, not regulatory approval. Thresholds are deliberately configurable and should be aligned with the portfolio, default definition, rating philosophy, materiality, and applicable governance.

## What is included

| Area | Methods |
|---|---|
| Discrimination | ROC AUC, normalized Gini, KS, stratified bootstrap Gini interval |
| Calibration | Observed/expected ratio, calibration intercept and slope, Brier score, log loss, quantile calibration table with Wilson intervals, Hosmer–Lemeshow |
| Stability | Numeric and categorical PSI, feature stability, score stability, relative Gini degradation |
| Time validation | Monthly/quarterly volumes, default rates, average PD, and Gini |
| Features | WoE transformation, information value, univariate Gini, Spearman/Pearson correlation, VIF, logistic coefficient Wald tests |
| Reporting | Tidy result table, traffic-light statuses, diagnostic tables, HTML/JSON/CSV export, CLI |

## Installation

```bash
python -m pip install git+https://github.com/qbaltabaev/pd-model-validation.git
```

For local development, install [uv](https://docs.astral.sh/uv/) and run:

```bash
git clone https://github.com/qbaltabaev/pd-model-validation.git
cd pd-model-validation
uv sync --all-extras --dev
uv run pytest
```

## Quick start

```python
import pandas as pd
from pd_model_validation import PDValidator

development = pd.read_csv("development.csv")
out_of_time = pd.read_csv("out_of_time.csv")

report = PDValidator().validate(
    development,
    current=out_of_time,
    target="default_12m",
    probability="predicted_pd",
    features=["utilization", "age_months", "arrears_count"],
    date="application_date",
)

print(report.to_frame())
print(report.tables["feature_stability"])
report.to_html("pd-validation-report.html")
```

Each result has a test name, value, traffic-light status, sample, scope, period, observation count, thresholds, and optional details. Supporting tables contain bin- and period-level evidence.

### WoE transformation

```python
from pd_model_validation import WoETransformer

transformer = WoETransformer(bins=10)
X_train_woe = transformer.fit_transform(X_train, y_train)
X_oot_woe = transformer.transform(X_oot)
print(transformer.iv_)
```

Breakpoints and mappings are fitted only on the reference sample. Missing values receive a dedicated bin; unseen categories receive neutral WoE, preventing target leakage into OOT data.

### Dummy data and fitted model

The repository includes [compact synthetic data](examples/data/dummy_pd_data.csv) with development and out-of-time samples. It contains no real customers, organizations, locations, currencies, or source-system identifiers. Run a complete preprocessing, logistic-regression fit, scoring, and validation workflow with:

```bash
uv run python examples/dummy_model.py
```

The example handles numeric and categorical features, missing values, unseen categories, and produces `dummy-pd-validation.html`. Its functions can also be imported in tests or notebooks.

### Command line

```bash
pdvalidate development.csv \
  --current out_of_time.csv \
  --target default_12m \
  --probability predicted_pd \
  --features utilization age_months arrears_count \
  --date application_date \
  --output validation.html
```

## Thresholds

Defaults are pragmatic starting points, not universal policy. Override them explicitly:

```python
from pd_model_validation import PDValidator, Thresholds, ValidationConfig

config = ValidationConfig(
    gini=Thresholds(green=0.45, amber=0.30, direction="higher"),
    psi=Thresholds(green=0.10, amber=0.20, direction="lower"),
)
validator = PDValidator(config)
```

See [methodology](docs/methodology.md) for definitions and interpretation, and [validation workflow](docs/validation-workflow.md) for a governance-oriented checklist. A fully reproducible synthetic example is in [examples/quickstart.py](examples/quickstart.py).

## Design principles

- Reference-sample transformations are reused unchanged on validation samples.
- Functions reject invalid targets and probabilities instead of silently clipping them.
- Missing pairs are handled consistently and all effective sample sizes are visible.
- Metrics are small, composable functions; the orchestrator is optional.
- Results are data frames and plain Python objects suitable for notebooks, pipelines, and audit evidence.
- The checked-in `uv.lock` and `.python-version` make development reproducible on Python 3.14 while the library remains tested across Python 3.10–3.14.

## License

MIT. See [LICENSE](LICENSE).
