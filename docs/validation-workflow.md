# Practical validation workflow

1. **Scope and definitions:** document use, portfolio, target horizon, default definition, exclusions, model version, and decision process.
2. **Data and implementation:** reconcile lineage, populations, filters, joins, missing values, transformations, coefficients, scaling, and production scores.
3. **Representativeness:** compare development, validation, OOT, and current populations; review selection bias and outcome maturity.
4. **Features:** assess rationale, leakage, data quality, WoE/IV, monotonicity, stability, univariate power, correlation, VIF, signs, and significance.
5. **Model performance:** assess discrimination, calibration, uncertainty, segments, vintages, and sensitivity to assumptions.
6. **Stability and monitoring:** review feature/score PSI, period metrics, overrides, concentrations, and trigger breaches.
7. **Limitations and decisions:** record findings, severity, compensating controls, remediation owner, due date, and approval.

Keep the raw result table and supporting bin/period tables as reproducible evidence. Explain every threshold and avoid treating a traffic light as the conclusion by itself.
