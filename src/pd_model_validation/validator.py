"""High-level orchestration for a complete PD model validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from ._validation import clean_binary_inputs, require_columns
from .calibration import calibration_summary, calibration_table, hosmer_lemeshow
from .discrimination import auc, gini, ks_statistic
from .features import correlation_diagnostics, feature_information_value, univariate_gini
from .report import ValidationReport
from .stability import (
    feature_stability,
    gini_degradation,
    performance_by_period,
    population_stability_index,
)
from .types import Status, Thresholds, ValidationResult


@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds and calculation settings for :class:`PDValidator`."""

    gini: Thresholds = field(default_factory=lambda: Thresholds(0.40, 0.25, "higher"))
    ks: Thresholds = field(default_factory=lambda: Thresholds(0.30, 0.20, "higher"))
    calibration_ratio_deviation: Thresholds = field(
        default_factory=lambda: Thresholds(0.10, 0.25, "lower")
    )
    hosmer_lemeshow_p: Thresholds = field(default_factory=lambda: Thresholds(0.05, 0.01, "higher"))
    psi: Thresholds = field(default_factory=lambda: Thresholds(0.10, 0.25, "lower"))
    gini_degradation: Thresholds = field(default_factory=lambda: Thresholds(0.10, 0.25, "lower"))
    feature_iv: Thresholds = field(default_factory=lambda: Thresholds(0.10, 0.02, "higher"))
    feature_correlation: Thresholds = field(default_factory=lambda: Thresholds(0.50, 0.70, "lower"))
    bins: int = 10
    min_period_observations: int = 30


class PDValidator:
    """Run model, feature, temporal, and sample-shift diagnostics.

    Parameters are column names, not fixed conventions. The reference sample
    normally represents development data; ``current`` is an OOT/OOS or recent
    monitoring sample.
    """

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    @staticmethod
    def _metric_result(
        test: str,
        value: float,
        thresholds: Thresholds | None,
        *,
        segment: str,
        n_obs: int,
        scope: str = "model",
        period: str | None = None,
        details: dict[str, object] | None = None,
    ) -> ValidationResult:
        return ValidationResult(
            test=test,
            value=float(value),
            status=thresholds.classify(value) if thresholds else Status.INFO,
            scope=scope,
            segment=segment,
            period=period,
            n_obs=n_obs,
            threshold_green=thresholds.green if thresholds else None,
            threshold_amber=thresholds.amber if thresholds else None,
            details=details or {},
        )

    def _sample_results(
        self, frame: pd.DataFrame, *, target: str, probability: str, segment: str
    ) -> tuple[list[ValidationResult], pd.DataFrame]:
        y_true, y_prob = clean_binary_inputs(frame[target], frame[probability])
        n_obs = len(y_true)
        summary = calibration_summary(y_true, y_prob)
        hl_statistic, hl_p_value, hl_df = hosmer_lemeshow(y_true, y_prob, n_bins=self.config.bins)
        results = [
            self._metric_result("auc", auc(y_true, y_prob), None, segment=segment, n_obs=n_obs),
            self._metric_result(
                "gini", gini(y_true, y_prob), self.config.gini, segment=segment, n_obs=n_obs
            ),
            self._metric_result(
                "ks", ks_statistic(y_true, y_prob), self.config.ks, segment=segment, n_obs=n_obs
            ),
            self._metric_result(
                "observed_default_rate",
                summary.observed_default_rate,
                None,
                segment=segment,
                n_obs=n_obs,
            ),
            self._metric_result("mean_pd", summary.mean_pd, None, segment=segment, n_obs=n_obs),
            self._metric_result(
                "brier_score", summary.brier_score, None, segment=segment, n_obs=n_obs
            ),
            self._metric_result("log_loss", summary.log_loss, None, segment=segment, n_obs=n_obs),
            self._metric_result(
                "calibration_intercept", summary.intercept, None, segment=segment, n_obs=n_obs
            ),
            self._metric_result(
                "calibration_slope", summary.slope, None, segment=segment, n_obs=n_obs
            ),
            self._metric_result(
                "observed_expected_ratio_deviation",
                abs(summary.observed_expected_ratio - 1.0),
                self.config.calibration_ratio_deviation,
                segment=segment,
                n_obs=n_obs,
                details={"observed_expected_ratio": summary.observed_expected_ratio},
            ),
            self._metric_result(
                "hosmer_lemeshow_p_value",
                hl_p_value,
                self.config.hosmer_lemeshow_p,
                segment=segment,
                n_obs=n_obs,
                details={"statistic": hl_statistic, "degrees_of_freedom": hl_df},
            ),
        ]
        return results, calibration_table(y_true, y_prob, n_bins=self.config.bins)

    def validate(
        self,
        reference: pd.DataFrame,
        *,
        target: str,
        probability: str,
        current: pd.DataFrame | None = None,
        features: list[str] | None = None,
        date: str | None = None,
    ) -> ValidationReport:
        """Run the validation and return tidy results with supporting tables."""
        features = features or []
        required = [target, probability, *features, *([date] if date else [])]
        require_columns(reference, required)
        if current is not None:
            require_columns(current, required)

        results, reference_calibration = self._sample_results(
            reference, target=target, probability=probability, segment="reference"
        )
        tables: dict[str, pd.DataFrame] = {"reference_calibration": reference_calibration}

        invalid_pd = int(
            (
                pd.to_numeric(reference[probability], errors="coerce").notna()
                & ~pd.to_numeric(reference[probability], errors="coerce").between(0, 1)
            ).sum()
        )
        results.append(
            self._metric_result(
                "invalid_probability_count",
                invalid_pd,
                None,
                segment="reference",
                n_obs=len(reference),
                scope="data_quality",
            )
        )

        if features:
            iv_table = feature_information_value(
                reference, features, target=target, bins=self.config.bins
            )
            standalone = univariate_gini(reference, features, target=target, bins=self.config.bins)
            tables["feature_information_value"] = iv_table
            tables["feature_univariate_gini"] = standalone
            for row in iv_table.itertuples(index=False):
                results.append(
                    self._metric_result(
                        "information_value",
                        cast(float, row.information_value),
                        self.config.feature_iv,
                        segment="reference",
                        n_obs=len(reference),
                        scope=str(row.feature),
                    )
                )
            numeric_features = [
                feature for feature in features if pd.api.types.is_numeric_dtype(reference[feature])
            ]
            if len(numeric_features) >= 2:
                correlations = correlation_diagnostics(reference, numeric_features)
                tables["feature_correlations"] = correlations
                for row in correlations.itertuples(index=False):
                    results.append(
                        self._metric_result(
                            "max_abs_correlation",
                            cast(float, row.max_abs_correlation),
                            self.config.feature_correlation,
                            segment="reference",
                            n_obs=len(reference),
                            scope=str(row.feature),
                            details={"correlated_with": row.correlated_with},
                        )
                    )

        if date:
            period_table = performance_by_period(
                reference,
                target=target,
                probability=probability,
                date=date,
                min_observations=self.config.min_period_observations,
            )
            tables["reference_performance_by_period"] = period_table
            for row in period_table.itertuples(index=False):
                results.append(
                    self._metric_result(
                        "period_gini",
                        cast(float, row.gini),
                        self.config.gini,
                        segment="reference",
                        period=str(row.period),
                        n_obs=cast(int, row.n),
                    )
                )

        if current is not None:
            current_results, current_calibration = self._sample_results(
                current, target=target, probability=probability, segment="current"
            )
            results.extend(current_results)
            tables["current_calibration"] = current_calibration
            score_psi = population_stability_index(
                reference[probability], current[probability], bins=self.config.bins
            )
            results.append(
                self._metric_result(
                    "score_psi",
                    score_psi,
                    self.config.psi,
                    segment="reference_vs_current",
                    n_obs=len(current),
                    scope="stability",
                )
            )
            degradation = gini_degradation(
                reference[target], reference[probability], current[target], current[probability]
            )
            results.append(
                self._metric_result(
                    "relative_gini_degradation",
                    degradation,
                    self.config.gini_degradation,
                    segment="reference_vs_current",
                    n_obs=len(current),
                    scope="stability",
                )
            )
            if features:
                feature_psi = feature_stability(reference, current, features, bins=self.config.bins)
                tables["feature_stability"] = feature_psi
                for row in feature_psi.itertuples(index=False):
                    results.append(
                        self._metric_result(
                            "feature_psi",
                            cast(float, row.psi),
                            self.config.psi,
                            segment="reference_vs_current",
                            n_obs=len(current),
                            scope=str(row.feature),
                        )
                    )
            if date:
                tables["current_performance_by_period"] = performance_by_period(
                    current,
                    target=target,
                    probability=probability,
                    date=date,
                    min_observations=self.config.min_period_observations,
                )

        return ValidationReport(results=results, tables=tables)
