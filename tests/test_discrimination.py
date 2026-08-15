import numpy as np
import pytest

from pd_model_validation import auc, bootstrap_gini, gini, ks_statistic


def test_perfect_discrimination() -> None:
    target = [0, 0, 1, 1]
    probability = [0.1, 0.2, 0.8, 0.9]
    assert auc(target, probability) == 1.0
    assert gini(target, probability) == 1.0
    assert ks_statistic(target, probability) == 1.0


def test_null_pairs_are_removed() -> None:
    assert gini([0, 1, 0, 1], [0.1, 0.9, np.nan, 0.8]) == 1.0


@pytest.mark.parametrize(
    ("target", "probability", "message"),
    [
        ([0, 2], [0.1, 0.9], "only 0 and 1"),
        ([0, 0], [0.1, 0.2], "both classes"),
        ([0, 1], [0.1], "same length"),
    ],
)
def test_invalid_inputs(target: list[int], probability: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gini(target, probability)


def test_bootstrap_gini_is_reproducible() -> None:
    estimate, lower, upper = bootstrap_gini(
        [0, 0, 0, 1, 1, 1], [0.1, 0.3, 0.4, 0.6, 0.8, 0.9], n_bootstrap=50, random_state=1
    )
    assert estimate == 1.0
    assert lower <= estimate <= upper


def test_bootstrap_gini_accepts_arbitrary_scores_like_auc() -> None:
    estimate, lower, upper = bootstrap_gini(
        [0, 0, 1, 1], [-2.0, -1.0, 1.0, 2.0], n_bootstrap=20, random_state=3
    )
    assert estimate == lower == upper == 1.0
