"""Generate the reproducible synthetic front-book underwriting dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

OUTPUT_PATH = Path(__file__).parent / "data" / "underwriting_pd_data.csv"


def generate_dataset(n_applications: int = 160, seed: int = 20260815) -> pd.DataFrame:
    """Create synthetic credit-card applications with a 12-month default outcome."""
    if not 100 <= n_applications <= 200:
        raise ValueError("n_applications must be between 100 and 200")

    rng = np.random.default_rng(seed)
    application_dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 1_095, n_applications)), unit="D"
    )
    sample = np.where(
        np.arange(n_applications) < int(n_applications * 0.75), "reference", "current"
    )

    annual_income = np.clip(rng.lognormal(np.log(68_000), 0.48, n_applications), 18_000, 260_000)
    employment_length = np.clip(rng.gamma(2.8, 2.6, n_applications), 0, 35)
    number_of_cards = np.clip(rng.poisson(2.4, n_applications), 0, 9)
    card_limit = np.where(
        number_of_cards > 0,
        np.clip(annual_income * rng.uniform(0.08, 0.38, n_applications), 1_000, 65_000),
        0,
    )
    utilization = np.where(
        number_of_cards > 0,
        np.clip(rng.beta(2.0, 2.8, n_applications) + rng.normal(0, 0.05, n_applications), 0, 1.2),
        0,
    )
    card_balance = card_limit * utilization

    has_mortgage = rng.random(n_applications) < 0.43
    mortgage_payment = np.where(
        has_mortgage, annual_income / 12 * rng.uniform(0.13, 0.31, n_applications), 0
    )
    has_auto_loan = rng.random(n_applications) < 0.56
    auto_payment = np.where(
        has_auto_loan, annual_income / 12 * rng.uniform(0.04, 0.12, n_applications), 0
    )
    total_loan_balance = (
        card_balance
        + mortgage_payment * rng.uniform(72, 180, n_applications)
        + auto_payment * rng.uniform(10, 48, n_applications)
        + rng.gamma(1.2, 2_500, n_applications)
    )
    deposit_balance = np.clip(
        rng.lognormal(np.log(np.maximum(annual_income * 0.12, 500)), 0.9), 0, 250_000
    )

    latent_risk = rng.normal(0, 1, n_applications) + 1.2 * utilization
    delinquencies_12m = np.clip(
        rng.poisson(np.exp(-1.45 + 0.45 * latent_risk), n_applications), 0, 7
    )
    delinquencies_6m = rng.binomial(delinquencies_12m, 0.62)
    delinquencies_3m = rng.binomial(delinquencies_6m, 0.52)
    bankruptcy_probability = expit(-3.7 + 0.75 * latent_risk + 0.2 * delinquencies_12m)
    bankruptcy_flag = rng.binomial(1, bankruptcy_probability)

    monthly_debt_service = (
        mortgage_payment + auto_payment + 0.025 * card_balance + 0.008 * total_loan_balance
    )
    debt_to_income = np.clip(monthly_debt_service / (annual_income / 12), 0, 1.5)
    recent_inquiries = np.clip(
        rng.poisson(1.0 + 0.45 * np.clip(latent_risk, 0, None), n_applications), 0, 9
    )
    oldest_credit_line = np.clip(rng.gamma(3.2, 34, n_applications), 3, 420)
    requested_limit = np.clip(
        annual_income / 12 * rng.uniform(0.5, 2.8, n_applications), 500, 35_000
    )
    employment_status = rng.choice(
        ["employed", "self_employed", "contract"],
        size=n_applications,
        p=[0.72, 0.18, 0.10],
    )
    housing_status = np.where(
        has_mortgage,
        "mortgage",
        rng.choice(["rent", "own"], size=n_applications, p=[0.72, 0.28]),
    )

    log_odds = (
        -3.25
        + 2.15 * utilization
        + 0.75 * delinquencies_3m
        + 0.24 * (delinquencies_12m - delinquencies_3m)
        + 1.55 * bankruptcy_flag
        + 1.05 * np.maximum(debt_to_income - 0.35, 0)
        + 0.13 * recent_inquiries
        - 0.012 * employment_length
        - 0.000012 * deposit_balance
    )
    pd90_12m = rng.binomial(1, expit(log_odds))

    data = pd.DataFrame(
        {
            "application_id": [f"APP{index:04d}" for index in range(1, n_applications + 1)],
            "sample": sample,
            "application_date": application_dates,
            "annual_income": annual_income,
            "employment_length_years": employment_length,
            "employment_status": employment_status,
            "housing_status": housing_status,
            "requested_credit_limit": requested_limit,
            "number_of_credit_cards": number_of_cards,
            "credit_card_balance": card_balance,
            "credit_card_utilization": utilization,
            "mortgage_payment": mortgage_payment,
            "total_loan_balance": total_loan_balance,
            "auto_loan_payment": auto_payment,
            "deposit_balance": deposit_balance,
            "delinquencies_last_3_months": delinquencies_3m,
            "delinquencies_last_6_months": delinquencies_6m,
            "delinquencies_last_12_months": delinquencies_12m,
            "recent_credit_inquiries": recent_inquiries,
            "months_since_oldest_credit_line": oldest_credit_line,
            "debt_to_income_ratio": debt_to_income,
            "bankruptcy_flag": bankruptcy_flag,
            "pd90_12m": pd90_12m,
        }
    )

    missing_deposits = rng.choice(data.index, size=5, replace=False)
    missing_income = rng.choice(data.index.difference(missing_deposits), size=3, replace=False)
    data.loc[missing_deposits, "deposit_balance"] = np.nan
    data.loc[missing_income, "annual_income"] = np.nan

    monetary_columns = [
        "annual_income",
        "requested_credit_limit",
        "credit_card_balance",
        "mortgage_payment",
        "total_loan_balance",
        "auto_loan_payment",
        "deposit_balance",
    ]
    data[monetary_columns] = data[monetary_columns].round(2)
    data["employment_length_years"] = data["employment_length_years"].round(1)
    data["credit_card_utilization"] = data["credit_card_utilization"].round(4)
    data["debt_to_income_ratio"] = data["debt_to_income_ratio"].round(4)
    data["months_since_oldest_credit_line"] = (
        data["months_since_oldest_credit_line"].round().astype(int)
    )
    return data


def main() -> int:
    """Regenerate the checked-in CSV from the fixed random seed."""
    data = generate_dataset()
    data.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(data)} applications to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
