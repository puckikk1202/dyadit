"""Real-real resampling keeps the evaluation reference fixed and sets disjoint."""

import numpy as np
import pytest

from dyadit.fid import frechet_distance
from dyadit.gt_fid import gt_trials


def test_gt_trials_match_direct_fid_and_are_reproducible():
    rng = np.random.default_rng(12)
    reference = rng.normal(size=(8, 4))
    candidates = rng.normal(size=(30, 4))
    result = gt_trials(reference, candidates, 8, 10, 223)
    repeated = gt_trials(reference, candidates, 8, 10, 223)
    np.testing.assert_array_equal(
        result["candidate_positions"], repeated["candidate_positions"]
    )
    np.testing.assert_array_equal(result["scores"], repeated["scores"])
    for selected, score in zip(result["candidate_positions"], result["scores"]):
        assert len(selected) == len(np.unique(selected)) == 8
        assert score == pytest.approx(frechet_distance(reference, candidates[selected]))
    assert result["mean"] == pytest.approx(np.mean(result["scores"]))
    assert result["std"] == pytest.approx(np.std(result["scores"], ddof=1))
    assert result["p025"] <= result["median"] <= result["p975"]


@pytest.mark.parametrize(
    "count,trials,pool_size", [(1, 10, 30), (8, 1, 30), (8, 10, 4), (7, 10, 30)]
)
def test_gt_trials_reject_unequal_or_insufficient_samples(count, trials, pool_size):
    with pytest.raises(ValueError):
        gt_trials(np.ones((8, 4)), np.ones((pool_size, 4)), count, trials, 223)
