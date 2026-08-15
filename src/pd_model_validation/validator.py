"""High-level orchestration for PD model validation and monitoring."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import cast

import numpy as np
import pandas as pd

from ._validation import clean_binary_inputs, require_columns
from ._version import __version__
from .calibration import (
    calibration_by_group,
    calibration_regression,
    calibration_summary,
    calibration_table,
    hosmer_lemeshow,
)
from .discrimination import auc, bootstrap_gini, ks_statistic
from .exceptions import InputValidationError
from .features import correlation_diagnostics, feature_information_value, univariate_gini
from .report import ValidationReport
from .stability import (
    feature_stability,
    gini_degradation,
    performance_by_period,
    population_stability_table,
)
from .types import Status, Thresholds, ValidationResult


@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds and calculation settings for :class:`PDValidator`.

    Default traffic lights are illustrative starting points, not policy or
    regulatory standards. Production users should instantiate and document a
    portfolio-specific configuration.
    """

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
    confidence_level: float = 0.95
    bootstrap_samples: int = 500
    random_state: int | None = 1729
    threshold_profile: str = "illustrative-defaults"

    def __post_init__(self) -> None:
        """Validate calculation settings before a run begins."""
        if self.bins < 2:
            raise ValueError("bins must be at least 2")
        if self.min_period_observations < 1:
            raise ValueError("min_period_observations must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not self.threshold_profile.strip():
            raise ValueError("threshold_profile must not be empty")


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Return an order-sensitive SHA-256 fingerprint for audit provenance."""
    hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


class PDValidator:
    """Run outcome validation, feature diagnostics, and sample monitoring.

    ``reference`` is a validation/reference population, not necessarily the
    model-development sample. A current sample may omit the target while its
    12-month outcome is immature; drift diagnostics remain available.
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
        feature: str | None = None,
        group: str | None = None,
        period: str | None = None,
        p_value: float | None = None,
        confidence_lower: float | None = None,
        confidence_upper: float | None = None,
        reason: str | None = None,
        status: Status | None = None,
        details: dict[str, object] | None = None,
    ) -> ValidationResult:
        effective_status = status
        if effective_status is None:
            effective_status = thresholds.classify(value) if thresholds else Status.INFO
        if not np.isfinite(value) and status is None:
            effective_status = Status.NOT_APPLICABLE
        return ValidationResult(
            test=test,
            value=float(value),
            status=effective_status,
            scope=scope,
            feature=feature,
            group=group,
            segment=segment,
            period=period,
            n_obs=n_obs,
            threshold_green=thresholds.green if thresholds else None,
            threshold_amber=thresholds.amber if thresholds else None,
            p_value=p_value,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            reason=reason,
            details=details or {},
        )

    def _data_quality_results(
        self, frame: pd.DataFrame, *, target: str, probability: str, segment: str
    ) -> list[ValidationResult]:
        """Validate raw binary inputs and report all excluded incomplete pairs."""
        raw_target = frame[target]
        raw_probability = frame[probability]
        numeric_target = pd.to_numeric(raw_target, errors="coerce")
        numeric_probability = pd.to_numeric(raw_probability, errors="coerce")
        target_missing = raw_target.isna()
        probability_missing = raw_probability.isna()
        invalid_target = ~target_missing & (
            ~np.isfinite(numeric_target) | ~numeric_target.isin([0, 1])
        )
        invalid_probability = ~probability_missing & (
            ~np.isfinite(numeric_probability) | ~numeric_probability.between(0, 1)
        )
        if invalid_target.any() or invalid_probability.any():
            raise InputValidationError(
                f"{segment} contains {int(invalid_target.sum())} invalid target values and "
                f"{int(invalid_probability.sum())} invalid probability values"
            )
        complete = ~(target_missing | probability_missing)
        return [
            self._metric_result(
                "input_row_count",
                len(frame),
                None,
                segment=segment,
                n_obs=len(frame),
                scope="data_quality",
            ),
            self._metric_result(
                "excluded_incomplete_pair_count",
                int((~complete).sum()),
                None,
                segment=segment,
                n_obs=len(frame),
                scope="data_quality",
                details={
                    "missing_target": int(target_missing.sum()),
                    "missing_probability": int(probability_missing.sum()),
                    "effective_observations": int(complete.sum()),
                },
            ),
        ]

    def _sample_results(
        self,
        frame: pd.DataFrame,
        *,
        target: str,
        probability: str,
        segment: str,
        allow_single_class: bool = False,
    ) -> tuple[list[ValidationResult], pd.DataFrame]:
        results = self._data_quality_results(
            frame, target=target, probability=probability, segment=segment
        )
        y_true, y_prob = clean_binary_inputs(
            frame[target], frame[probability], allow_single_class=allow_single_class
        )
        n_obs = len(y_true)
        has_both_classes = len(np.unique(y_true)) == 2
        summary = calibration_summary(
            y_true,
            y_prob,
            confidence=self.config.confidence_level,
            raise_on_nonconvergence=False,
            allow_single_class=allow_single_class,
        )
        if has_both_classes:
            regression = calibration_regression(y_true, y_prob, raise_on_nonconvergence=False)
            gini_value, gini_lower, gini_upper = bootstrap_gini(
                y_true,
                y_prob,
                confidence=self.config.confidence_level,
                n_bootstrap=self.config.bootstrap_samples,
                random_state=self.config.random_state,
            )
            auc_value = auc(y_true, y_prob)
            ks_value = ks_statistic(y_true, y_prob)
            hl_statistic, hl_p_value, hl_df = hosmer_lemeshow(
                y_true, y_prob, n_bins=self.config.bins
            )
        else:
            regression = None
            auc_value = gini_value = gini_lower = gini_upper = ks_value = np.nan
            hl_statistic = hl_p_value = np.nan
            hl_df = 0
        discrimination_reason = None if has_both_classes else "sample contains only one class"
        ratio_deviation = abs(summary.observed_expected_ratio - 1.0)
        ratio_status = Status.RED if not np.isfinite(ratio_deviation) else None
        results.extend(
            [
                self._metric_result(
                    "auc",
                    auc_value,
                    None,
                    segment=segment,
                    n_obs=n_obs,
                    reason=discrimination_reason,
                ),
                self._metric_result(
                    "gini",
                    gini_value,
                    self.config.gini,
                    segment=segment,
                    n_obs=n_obs,
                    confidence_lower=gini_lower,
                    confidence_upper=gini_upper,
                    reason=discrimination_reason,
                    details={"bootstrap_samples": self.config.bootstrap_samples},
                ),
                self._metric_result(
                    "ks",
                    ks_value,
                    self.config.ks,
                    segment=segment,
                    n_obs=n_obs,
                    reason=discrimination_reason,
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
                self._metric_result(
                    "log_loss", summary.log_loss, None, segment=segment, n_obs=n_obs
                ),
                self._metric_result(
                    "calibration_intercept",
                    summary.intercept,
                    None,
                    segment=segment,
                    n_obs=n_obs,
                    status=Status.NOT_APPLICABLE
                    if regression is None or not regression.converged
                    else None,
                    reason=(
                        "sample contains only one class"
                        if regression is None
                        else "calibration regression did not converge"
                        if not regression.converged
                        else None
                    ),
                    details={
                        "std_error": regression.intercept_std_error if regression else np.nan,
                        "converged": regression.converged if regression else False,
                        "iterations": regression.iterations if regression else 0,
                    },
                ),
                self._metric_result(
                    "calibration_slope",
                    summary.slope,
                    None,
                    segment=segment,
                    n_obs=n_obs,
                    status=Status.NOT_APPLICABLE
                    if regression is None or not regression.converged
                    else None,
                    reason=(
                        "sample contains only one class"
                        if regression is None
                        else "calibration regression did not converge"
                        if not regression.converged
                        else "constant predictions; slope is not identifiable"
                        if not np.isfinite(summary.slope)
                        else None
                    ),
                    details={"std_error": regression.slope_std_error if regression else np.nan},
                ),
                self._metric_result(
                    "observed_expected_ratio",
                    summary.observed_expected_ratio,
                    None,
                    segment=segment,
                    n_obs=n_obs,
                    confidence_lower=summary.observed_expected_ratio_lower,
                    confidence_upper=summary.observed_expected_ratio_upper,
                    reason="mean predicted PD is zero"
                    if not np.isfinite(summary.observed_expected_ratio)
                    else None,
                ),
                self._metric_result(
                    "observed_expected_ratio_deviation",
                    ratio_deviation,
                    self.config.calibration_ratio_deviation,
                    segment=segment,
                    n_obs=n_obs,
                    status=ratio_status,
                    reason="mean predicted PD is zero" if ratio_status is Status.RED else None,
                    details={"observed_expected_ratio": summary.observed_expected_ratio},
                ),
                self._metric_result(
                    "hosmer_lemeshow_p_value",
                    hl_p_value,
                    self.config.hosmer_lemeshow_p,
                    segment=segment,
                    n_obs=n_obs,
                    p_value=hl_p_value,
                    details={"statistic": hl_statistic, "degrees_of_freedom": hl_df},
                ),
            ]
        )
        table = calibration_table(
            y_true,
            y_prob,
            n_bins=self.config.bins,
            confidence=self.config.confidence_level,
            allow_single_class=allow_single_class,
        )
        return results, table

    def _period_results(self, table: pd.DataFrame, *, segment: str) -> list[ValidationResult]:
        """Convert a temporal evidence table into classified findings."""
        results = []
        for row in table.itertuples(index=False):
            metric = cast(float, row.gini)
            reason = None
            if not np.isfinite(metric):
                reason = "insufficient observations or only one observed class"
            results.append(
                self._metric_result(
                    "period_gini",
                    metric,
                    self.config.gini,
                    segment=segment,
                    period=str(row.period),
                    n_obs=cast(int, row.n),
                    reason=reason,
                )
            )
        return results

    def _group_results(self, table: pd.DataFrame, *, segment: str) -> list[ValidationResult]:
        """Convert grouped calibration evidence into classified O/E findings."""
        results = []
        for row in table.itertuples(index=False):
            ratio = cast(float, row.observed_expected_ratio)
            deviation = abs(ratio - 1.0)
            status = Status.RED if not np.isfinite(deviation) else None
            results.append(
                self._metric_result(
                    "group_observed_expected_ratio_deviation",
                    deviation,
                    self.config.calibration_ratio_deviation,
                    segment=segment,
                    n_obs=cast(int, row.n),
                    scope="group",
                    group=str(row.group),
                    p_value=cast(float, row.p_value),
                    status=status,
                    reason="expected defaults are zero" if status is Status.RED else None,
                    details={
                        "defaults": cast(int, row.defaults),
                        "expected_defaults": cast(float, row.expected_defaults),
                        "observed_expected_ratio": ratio,
                    },
                )
            )
        return results

    def _drift_results(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        probability: str,
        features: list[str],
    ) -> tuple[list[ValidationResult], dict[str, pd.DataFrame]]:
        """Calculate score and feature drift without requiring outcomes."""
        for frame, segment in ((reference, "reference"), (current, "current")):
            raw_probability = frame[probability]
            numeric_probability = pd.to_numeric(raw_probability, errors="coerce")
            invalid = raw_probability.notna() & (
                ~np.isfinite(numeric_probability) | ~numeric_probability.between(0, 1)
            )
            if invalid.any():
                raise InputValidationError(
                    f"{segment} contains {int(invalid.sum())} invalid probability values"
                )
        score_table = population_stability_table(
            reference[probability], current[probability], bins=self.config.bins
        )
        score_psi = float(score_table["psi_contribution"].sum())
        results = [
            self._metric_result(
                "score_psi",
                score_psi,
                self.config.psi,
                segment="reference_vs_current",
                n_obs=len(current),
                scope="stability",
                details={
                    "reference_missing": int(reference[probability].isna().sum()),
                    "current_missing": int(current[probability].isna().sum()),
                },
            )
        ]
        tables = {"score_stability": score_table}
        if features:
            feature_psi = feature_stability(reference, current, features, bins=self.config.bins)
            tables["feature_stability"] = feature_psi
            contributions = []
            for feature in features:
                table = population_stability_table(
                    reference[feature], current[feature], bins=self.config.bins
                ).assign(feature=feature)
                contributions.append(table)
            tables["feature_stability_contributions"] = pd.concat(contributions, ignore_index=True)
            for row in feature_psi.itertuples(index=False):
                results.append(
                    self._metric_result(
                        "feature_psi",
                        cast(float, row.psi),
                        self.config.psi,
                        segment="reference_vs_current",
                        n_obs=len(current),
                        scope="feature",
                        feature=str(row.feature),
                    )
                )
        return results, tables

    def _metadata(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame | None,
        *,
        target: str | None,
        probability: str,
        features: list[str],
        date: str | None,
        group: str | None,
        context: Mapping[str, object] | None,
    ) -> dict[str, object]:
        """Build reproducibility metadata for each validation run."""
        samples: dict[str, object] = {
            "reference": {
                "rows": len(reference),
                "columns": len(reference.columns),
                "sha256": _frame_fingerprint(reference),
            }
        }
        if current is not None:
            samples["current"] = {
                "rows": len(current),
                "columns": len(current.columns),
                "sha256": _frame_fingerprint(current),
                "outcome_available": bool(
                    target and target in current.columns and current[target].notna().any()
                ),
            }
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_version": __version__,
            "configuration": asdict(self.config),
            "columns": {
                "target": target,
                "probability": probability,
                "features": features,
                "date": date,
                "group": group,
            },
            "samples": samples,
            "context": dict(context or {}),
        }

    def monitor_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        probability: str,
        features: list[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ValidationReport:
        """Monitor score and feature drift when current outcomes are unavailable."""
        selected_features = features or []
        require_columns(reference, [probability, *selected_features])
        require_columns(current, [probability, *selected_features])
        results, tables = self._drift_results(
            reference, current, probability=probability, features=selected_features
        )
        run_metadata = self._metadata(
            reference,
            current,
            target=None,
            probability=probability,
            features=selected_features,
            date=None,
            group=None,
            context=metadata,
        )
        return ValidationReport(results=results, tables=tables, metadata=run_metadata)

    def validate(
        self,
        reference: pd.DataFrame,
        *,
        target: str,
        probability: str,
        current: pd.DataFrame | None = None,
        features: list[str] | None = None,
        date: str | None = None,
        group: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ValidationReport:
        """Validate mature outcomes and optionally monitor a current sample.

        The current sample may omit ``target`` while outcomes are immature. In
        that case, score and feature stability are produced and outcome-based
        current metrics are skipped explicitly.
        """
        selected_features = features or []
        reference_required = [
            target,
            probability,
            *selected_features,
            *([date] if date else []),
            *([group] if group else []),
        ]
        require_columns(reference, reference_required)
        if current is not None:
            current_required = [
                probability,
                *selected_features,
                *([date] if date else []),
                *([group] if group else []),
            ]
            require_columns(current, current_required)

        results, reference_calibration = self._sample_results(
            reference, target=target, probability=probability, segment="reference"
        )
        tables: dict[str, pd.DataFrame] = {"reference_calibration": reference_calibration}

        if group:
            group_table = calibration_by_group(
                reference[target], reference[probability], reference[group]
            )
            tables["reference_calibration_by_group"] = group_table
            results.extend(self._group_results(group_table, segment="reference"))

        if selected_features:
            iv_table = feature_information_value(
                reference, selected_features, target=target, bins=self.config.bins
            )
            standalone = univariate_gini(
                reference, selected_features, target=target, bins=self.config.bins
            )
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
                        scope="feature",
                        feature=str(row.feature),
                        reason="apparent in-sample screening statistic",
                    )
                )
            numeric_features = [
                feature
                for feature in selected_features
                if pd.api.types.is_numeric_dtype(reference[feature])
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
                            scope="feature",
                            feature=str(row.feature),
                            reason="constant or insufficient paired observations"
                            if pd.isna(row.max_abs_correlation)
                            else None,
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
            results.extend(self._period_results(period_table, segment="reference"))

        if current is not None:
            drift_results, drift_tables = self._drift_results(
                reference, current, probability=probability, features=selected_features
            )
            results.extend(drift_results)
            tables.update(drift_tables)
            current_outcome_available = target in current.columns and current[target].notna().any()
            if current_outcome_available:
                current_results, current_calibration = self._sample_results(
                    current,
                    target=target,
                    probability=probability,
                    segment="current",
                    allow_single_class=True,
                )
                results.extend(current_results)
                tables["current_calibration"] = current_calibration
                if group:
                    current_group_table = calibration_by_group(
                        current[target], current[probability], current[group]
                    )
                    tables["current_calibration_by_group"] = current_group_table
                    results.extend(self._group_results(current_group_table, segment="current"))
                current_classes = pd.to_numeric(current[target], errors="coerce").dropna().nunique()
                degradation = (
                    gini_degradation(
                        reference[target],
                        reference[probability],
                        current[target],
                        current[probability],
                    )
                    if current_classes == 2
                    else np.nan
                )
                results.append(
                    self._metric_result(
                        "relative_gini_degradation",
                        degradation,
                        self.config.gini_degradation,
                        segment="reference_vs_current",
                        n_obs=len(current),
                        scope="stability",
                        reason="current sample contains only one class"
                        if current_classes != 2
                        else None,
                    )
                )
                if date:
                    current_period_table = performance_by_period(
                        current,
                        target=target,
                        probability=probability,
                        date=date,
                        min_observations=self.config.min_period_observations,
                    )
                    tables["current_performance_by_period"] = current_period_table
                    results.extend(self._period_results(current_period_table, segment="current"))
            else:
                results.append(
                    self._metric_result(
                        "current_outcome_available",
                        0.0,
                        None,
                        segment="current",
                        n_obs=len(current),
                        scope="data_quality",
                        reason="current outcome is absent or not yet mature; outcome tests skipped",
                    )
                )

        run_metadata = self._metadata(
            reference,
            current,
            target=target,
            probability=probability,
            features=selected_features,
            date=date,
            group=group,
            context=metadata,
        )
        return ValidationReport(results=results, tables=tables, metadata=run_metadata)
