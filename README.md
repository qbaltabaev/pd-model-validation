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
    target="pd90_12m",
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

### Front-book underwriting dataset and model

The repository includes [160 synthetic credit-card applications](examples/data/underwriting_pd_data.csv) across development and out-of-time periods. The `pd90_12m` outcome identifies applications that reached at least 90 days past due within 12 months. Features cover income, employment, requested limit, card exposure and utilization, mortgage and auto payments, total loan and deposit balances, recent delinquencies, credit history, indebtedness, inquiries, and bankruptcy. See the [data dictionary and modeling frame](docs/example-data.md) for precise definitions.

The data represents a front-book underwriting use case and contains no real customers, organizations, locations, currencies, or source-system identifiers. Regenerate it deterministically or run the fitted model and complete validation with:

```bash
uv run python examples/generate_underwriting_data.py
uv run python examples/underwriting_model.py
```

The model uses reference-only preprocessing and logistic regression, scores both samples, and produces `underwriting-pd-validation.html`. Its functions can also be imported in tests or notebooks.

### Command line

```bash
pdvalidate development.csv \
  --current out_of_time.csv \
  --target pd90_12m \
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
