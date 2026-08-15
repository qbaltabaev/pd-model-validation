import numpy as np
import pandas as pd
import pytest

from pd_model_validation import (
    WoETransformer,
    correlation_diagnostics,
    feature_information_value,
    information_value,
    logistic_coefficient_test,
    univariate_gini,
    variance_inflation_factors,
    woe_table,
)


def test_woe_table_is_finite() -> None:
    table = woe_table(pd.Series(["a", "a", "b", "b"]), pd.Series([0, 0, 1, 1]))
    assert np.isfinite(table["woe"]).all()
    assert table["iv_component"].sum() > 0


def test_woe_rejects_invalid_settings_and_targets() -> None:
    with pytest.raises(ValueError, match="smoothing"):
        woe_table(pd.Series(["a", "b"]), pd.Series([0, 1]), smoothing=0)
    with pytest.raises(ValueError, match="only 0 and 1"):
        woe_table(pd.Series(["a", "b"]), pd.Series([0, 2]))
    with pytest.raises(ValueError, match="both classes"):
        woe_table(pd.Series(["a", "b"]), pd.Series([0, 0]))


def test_transformer_reuses_bins_and_handles_unknown(reference) -> None:
    transformer = WoETransformer(bins=5).fit(reference[["x1", "category"]], reference["target"])
    transformed = transformer.transform(
        pd.DataFrame({"x1": [0.0, np.nan], "category": ["new", None]})
    )
    assert list(transformed.columns) == ["x1_woe", "category_woe"]
    assert transformed.notna().all().all()
    assert transformed.loc[0, "category_woe"] == 0.0
    assert set(transformer.iv_) == {"x1", "category"}

    strict = WoETransformer(bins=5, handle_unknown="error").fit(
        reference[["category"]], reference["target"]
    )
    with pytest.raises(ValueError, match="unknown bins"):
        strict.transform(pd.DataFrame({"category": ["new"]}))


def test_unfitted_transformer_raises(reference) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        WoETransformer().transform(reference[["x1"]])
    with pytest.raises(RuntimeError, match="not fitted"):
        WoETransformer().get_feature_names_out()


@pytest.mark.parametrize(
    ("transformer", "message"),
    [
        (WoETransformer(bins=1), "bins"),
        (WoETransformer(smoothing=0), "smoothing"),
        (WoETransformer(handle_unknown="invalid"), "handle_unknown"),  # type: ignore[arg-type]
    ],
)
def test_transformer_rejects_invalid_configuration(reference, transformer, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        transformer.fit(reference[["x1"]], reference["target"])


def test_transformer_feature_names_must_match_fit(reference) -> None:
    transformer = WoETransformer().fit(reference[["x1"]], reference["target"])
    assert transformer.get_feature_names_out().tolist() == ["x1_woe"]
    with pytest.raises(ValueError, match="must match"):
        transformer.get_feature_names_out(["x2"])


def test_feature_diagnostics(reference) -> None:
    assert information_value(reference["x1"], reference["target"]) > 0
    iv = feature_information_value(reference, ["x1", "x2"], target="target")
    standalone = univariate_gini(reference, ["x1", "x2"], target="target")
    correlations = correlation_diagnostics(reference, ["x1", "x2"])
    assert len(iv) == len(standalone) == len(correlations) == 2
    assert (standalone["gini"] >= 0).all()


def test_vif_and_coefficient_tests(reference) -> None:
    vif = variance_inflation_factors(reference, ["x1", "x2"])
    coefficients = logistic_coefficient_test(reference, ["x1", "x2"], target="target")
    assert np.allclose(vif["vif"], 1, atol=0.05)
    assert list(coefficients["feature"]) == ["intercept", "x1", "x2"]
    assert coefficients["p_value"].between(0, 1).all()


def test_coefficient_test_rejects_rank_deficiency_and_iteration_errors(reference) -> None:
    reference["duplicate"] = reference["x1"]
    with pytest.raises(ValueError, match="rank deficient"):
        logistic_coefficient_test(reference, ["x1", "duplicate"], target="target")
    with pytest.raises(ValueError, match="max_iter"):
        logistic_coefficient_test(reference, ["x1"], target="target", max_iter=0)


def test_vif_rejects_constant(reference) -> None:
    reference["constant"] = 1
    with pytest.raises(ValueError, match="constant"):
        variance_inflation_factors(reference, ["x1", "constant"])


def test_vif_flags_exact_multicollinearity() -> None:
    frame = pd.DataFrame({"x1": np.arange(20.0), "x2": np.arange(20.0) * 2})
    result = variance_inflation_factors(frame, ["x1", "x2"])
    assert np.isinf(result["vif"]).all()


def test_correlation_returns_na_for_constant_feature(reference) -> None:
    reference["constant"] = 1.0
    result = correlation_diagnostics(reference, ["x1", "constant"])
    constant = result.loc[result["feature"] == "constant"].iloc[0]
    assert np.isnan(constant["max_abs_correlation"])
    assert constant["correlated_with"] is None
