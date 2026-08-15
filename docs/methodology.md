# Methodology reference

This document states the definitions implemented by the package and the main interpretation limits. It is intentionally concise; validation conclusions should also consider portfolio context and qualitative evidence.

## Discrimination

- **AUC** is the probability that a randomly selected default receives a higher predicted PD than a randomly selected non-default.
- **Gini** is `2 × AUC − 1`.
- **KS** is the maximum vertical distance between the empirical score distributions of defaults and non-defaults.
- **Bootstrap Gini interval** resamples defaults and non-defaults independently, preserving class representation. `PDValidator` records the interval, confidence level, number of resamples, and deterministic seed.

Discrimination does not establish calibration. Confidence intervals and period-level results matter when defaults are sparse.

## Calibration

- **Observed/expected (O/E)** is mean observed default divided by mean predicted PD. The validator reports a Wilson interval for the observed rate translated to the O/E scale. Zero expected defaults with observed events is a RED data/model failure, not N/A.
- **Calibration intercept and slope** come from `logit(Y) = a + b × logit(PD)`. Ideal values are 0 and 1. A slope below 1 often indicates overly extreme predictions. Standard errors and convergence diagnostics are retained. Non-converged fits are explicitly N/A in the orchestrated report.
- **Brier score** is mean squared probability error. **Log loss** is the Bernoulli negative log likelihood.
- **Calibration tables** target equal-frequency bins, preserve tied PD values, and include configurable Wilson intervals. Consequently, the effective number of bins can be lower than requested and is stored in table attributes.
- **Grouped calibration** compares observed and expected defaults by rating grade, pool, or segment. Its outcome test uses a normal approximation to the Poisson-binomial distribution and should be treated cautiously for sparse groups.
- **Hosmer–Lemeshow** compares observed and expected defaults by quantile bin. Its p-value is sample-size sensitive and must not be the sole calibration conclusion.

## Stability and time

Population Stability Index is

`PSI = Σ (actual_share − expected_share) × ln(actual_share / expected_share)`.

Numeric breakpoints are fitted on the reference sample; categorical levels are aligned by union. Missing values form an explicit `<MISSING>` bucket. `population_stability_table` returns bucket counts, normalized shares, and contributions so a scalar PSI remains traceable. A common convention is PSI below 0.10 (stable), 0.10–0.25 (review), and above 0.25 (material shift), but portfolio policy takes precedence.

Period tables show volume, defaults, observed rate, average PD, and Gini. Low-volume or single-class periods return missing Gini rather than a misleading number.

## WoE and feature diagnostics

WoE uses the scorecard convention `ln(% non-events / % events)`. Information value sums `(% non-events − % events) × WoE` across bins. Additive smoothing prevents infinite values. Quantile breakpoints and mappings are learned on reference data only.

Univariate WoE/Gini fitted and measured on one sample is an apparent screening statistic, not an unbiased performance estimate or proof of incremental contribution. Use a holdout mapping or cross-fitting when estimating performance. Review feature meaning, missingness, overrides, monotonicity, temporal stability, coefficient sign and significance, correlation, and VIF together. Exact linear dependence returns infinite VIF; constant features have undefined correlation/VIF.

## Input and edge-case conventions

- Targets must be binary `0/1`; PDs used for calibration must be finite and within `[0, 1]`.
- Invalid non-null inputs raise `InputValidationError` before metrics run. Missing pairs are excluded and counted in report evidence.
- Discrimination-only functions accept arbitrary ordered scores; probability-based functions remain strict.
- A current monitoring sample may omit an immature outcome. PSI continues to run, while outcome tests are explicitly skipped.
- Low-volume or single-class periods produce N/A Gini with a reason. Constant predictions produce an identifiable calibration intercept but no slope.
- Strict JSON export converts non-finite numerical results to `null`.

## Important limitations

- The toolkit does not choose a default definition, performance window, observation window, censoring treatment, or sampling design.
- It does not replace representativeness analysis, data lineage, implementation verification, expert judgement, or model governance.
- Naive observed default rates can be biased for immature vintages. Construct a mature outcome sample before using these tests.
- Multiple testing, sparse defaults, rejected applicants, overrides, and economic-cycle coverage require explicit treatment outside a single headline metric.

## Supervisory context

The package is not a regulatory rules engine. Where relevant, methodology and
sample design should be reconciled with the current [Basel Framework CRE36](https://www.bis.org/basel_framework/chapter/CRE/36.htm)
requirements for PD estimation, long-run evidence, representativeness, and
review, as well as [Basel Framework CRI30](https://www.bis.org/basel_framework/chapter/CRI/30.htm)
for internal-ratings systems and low-default portfolios. Applicable local rules
and internal policy take precedence over illustrative package defaults.
