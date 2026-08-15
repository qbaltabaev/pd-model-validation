# Practical validation workflow

1. **Scope and definitions:** document use, portfolio, target horizon, default definition, exclusions, model version, and decision process.
2. **Data and implementation:** reconcile lineage, populations, filters, joins, missing values, transformations, coefficients, scaling, production scores, and an independently reproduced score sample.
3. **Representativeness:** compare development, validation, OOT, and current populations; review selection bias and outcome maturity.
4. **Features:** assess rationale, leakage, data quality, WoE/IV, monotonicity, stability, univariate power, correlation, VIF, signs, and significance.
5. **Model performance:** assess discrimination, aggregate and grade/pool calibration, uncertainty, segments, vintages, benchmarks, and sensitivity to assumptions.
6. **Stability and monitoring:** review feature/score PSI, period metrics, overrides, concentrations, and trigger breaches.
7. **Limitations and decisions:** record findings, severity, compensating controls, remediation owner, due date, and approval.

Keep the raw result table and supporting bin/period tables as reproducible evidence. Explain every threshold and avoid treating a traffic light as the conclusion by itself.

For every reproducible run, retain the model identifier and version, code
revision, package version, configuration, sample definitions, row exclusions,
data fingerprints, execution time, limitations, reviewer, and decision. The
report captures technical provenance supplied to `metadata`; organizational
approvals and remediation ownership remain outside the library.
