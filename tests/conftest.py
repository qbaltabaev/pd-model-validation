from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _sample(seed: int, size: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(shift, 1, size)
    x2 = rng.normal(0, 1, size)
    category = rng.choice(["a", "b", "c"], size=size, p=[0.5, 0.3, 0.2])
    probability = 1 / (1 + np.exp(-(-1.8 + 1.1 * x1 - 0.4 * x2 + (category == "c") * 0.5)))
    target = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=size, freq="D"),
            "x1": x1,
            "x2": x2,
            "category": category,
            "pd": probability,
            "target": target,
        }
    )


@pytest.fixture
def reference() -> pd.DataFrame:
    return _sample(42, 500)


@pytest.fixture
def current() -> pd.DataFrame:
    return _sample(84, 350, shift=0.4)
