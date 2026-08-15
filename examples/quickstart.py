"""Run a complete validation on reproducible synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pd_model_validation import PDValidator


def make_sample(seed: int, observations: int, *, shift: float = 0.0) -> pd.DataFrame:
    """Generate a reproducible synthetic sample for the quick-start example."""
    rng = np.random.default_rng(seed)
    utilization = rng.beta(2 + shift, 5, observations)
    age_months = rng.gamma(3, 18, observations)
    arrears_count = rng.poisson(0.3 + shift, observations)
    log_odds = -3.2 + 2.4 * utilization - 0.005 * age_months + 0.45 * arrears_count
    pd_estimate = 1 / (1 + np.exp(-log_odds))
    target = rng.binomial(1, pd_estimate)
    return pd.DataFrame(
        {
            "application_date": pd.date_range("2022-01-01", periods=observations, freq="D"),
            "utilization": utilization,
            "age_months": age_months,
            "arrears_count": arrears_count,
            "predicted_pd": pd_estimate,
            "default_12m": target,
        }
    )


if __name__ == "__main__":
    development = make_sample(7, 1_500)
    out_of_time = make_sample(11, 900, shift=0.15)
    validation = PDValidator().validate(
        development,
        current=out_of_time,
        target="default_12m",
        probability="predicted_pd",
        features=["utilization", "age_months", "arrears_count"],
        date="application_date",
    )
    print(validation.to_frame().to_string(index=False))
    validation.to_html("pd-validation-report.html")
