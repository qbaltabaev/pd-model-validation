# Methodology reference

This document states the definitions implemented by the package and the main interpretation limits. It is intentionally concise; validation conclusions should also consider portfolio context and qualitative evidence.

## Discrimination

- **AUC** is the probability that a randomly selected default receives a higher predicted PD than a randomly selected non-default.
- **Gini** is `2 × AUC − 1`.
- **KS** is the maximum vertical distance between the empirical score distributions of defaults and non-defaults.
- **Bootstrap Gini interval** resamples defaults and non-defaults independently, preserving class representation.

Discrimination does not establish calibration. Confidence intervals and period-level results matter when defaults are sparse.

## Calibration

- **Observed/expected (O/E)** is mean observed default divided by mean predicted PD.
- **Calibration intercept and slope** come from `logit(Y) = a + b × logit(PD)`. Ideal values are 0 and 1. A slope below 1 often indicates overly extreme predictions.
- **Brier score** is mean squared probability error. **Log loss** is the Bernoulli negative log likelihood.
- **Calibration tables** use equal-frequency bins and include 95% Wilson intervals for observed default rates.
- **Hosmer–Lemeshow** compares observed and expected defaults by quantile bin. Its p-value is sample-size sensitive and must not be the sole calibration conclusion.

## Stability and time

Population Stability Index is

`PSI = Σ (actual_share − expected_share) × ln(actual_share / expected_share)`.

Numeric breakpoints are fitted on the reference sample; categorical levels are aligned by union. Missing values form a separate bucket. A common convention is PSI below 0.10 (stable), 0.10–0.25 (review), and above 0.25 (material shift), but portfolio policy takes precedence.

Period tables show volume, defaults, observed rate, average PD, and Gini. Low-volume or single-class periods return missing Gini rather than a misleading number.

## WoE and feature diagnostics

WoE uses the scorecard convention `ln(% non-events / % events)`. Information value sums `(% non-events − % events) × WoE` across bins. Additive smoothing prevents infinite values. Quantile breakpoints and mappings are learned on reference data only.

Univariate measures are screening diagnostics, not proof of incremental contribution. Review feature meaning, missingness, overrides, monotonicity, temporal stability, coefficient sign and significance, correlation, and VIF together.

## Important limitations

- The toolkit does not choose a default definition, performance window, observation window, censoring treatment, or sampling design.
- It does not replace representativeness analysis, data lineage, implementation verification, expert judgement, or model governance.
- Naive observed default rates can be biased for immature vintages. Construct a mature outcome sample before using these tests.
- Multiple testing, sparse defaults, rejected applicants, overrides, and economic-cycle coverage require explicit treatment outside a single headline metric.
