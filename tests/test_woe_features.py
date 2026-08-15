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


def test_transformer_reuses_bins_and_handles_unknown(reference) -> None:
    transformer = WoETransformer(bins=5).fit(reference[["x1", "category"]], reference["target"])
    transformed = transformer.transform(
        pd.DataFrame({"x1": [0.0, np.nan], "category": ["new", None]})
    )
    assert list(transformed.columns) == ["x1_woe", "category_woe"]
    assert transformed.notna().all().all()
    assert transformed.loc[0, "category_woe"] == 0.0
    assert set(transformer.iv_) == {"x1", "category"}


def test_unfitted_transformer_raises(reference) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        WoETransformer().transform(reference[["x1"]])


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


def test_vif_rejects_constant(reference) -> None:
    reference["constant"] = 1
    with pytest.raises(ValueError, match="constant"):
        variance_inflation_factors(reference, ["x1", "constant"])
