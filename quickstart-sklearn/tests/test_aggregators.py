import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearnexample.aggregators import (  # noqa: E402
    aggregate,
    coordinate_median,
    fedavg,
    flatten_params,
    krum,
    trimmed_mean,
    unflatten_params,
)
from sklearnexample.attacks import flip_labels, num_malicious, poison_update  # noqa: E402


@pytest.fixture
def updates():
    """8 honest clients near +1, 2 attackers at -50 (20 % malicious)."""
    rng = np.random.default_rng(0)
    honest = 1.0 + 0.05 * rng.standard_normal((8, 6))
    attackers = np.full((2, 6), -50.0)
    return np.vstack([honest, attackers])


def test_fedavg_is_broken_by_attackers(updates):
    agg = fedavg(updates, np.ones(len(updates)))
    assert agg.mean() < -5  # dragged far from the honest value of ~1


def test_median_resists(updates):
    assert np.allclose(coordinate_median(updates), 1.0, atol=0.2)


def test_trimmed_mean_resists(updates):
    assert np.allclose(trimmed_mean(updates, 0.2), 1.0, atol=0.2)


def test_trimmed_mean_zero_ratio_is_mean(updates):
    assert np.allclose(trimmed_mean(updates, 0.0), updates.mean(axis=0))


def test_krum_selects_honest(updates):
    agg, selected = krum(updates, f=2, m=1)
    assert selected[0] < 8
    assert np.allclose(agg, 1.0, atol=0.3)


def test_multikrum_excludes_attackers(updates):
    agg, selected = aggregate("multikrum", updates, np.ones(10), f=2)
    assert set(selected).isdisjoint({8, 9})
    assert np.allclose(agg, 1.0, atol=0.2)


def test_krum_f_is_clamped_for_small_federations():
    u = np.random.default_rng(1).standard_normal((4, 3))
    agg, selected = krum(u, f=3, m=1)  # 4 < 2*3+3, must not crash
    assert agg.shape == (3,) and len(selected) == 1


def test_unknown_method():
    with pytest.raises(ValueError):
        aggregate("nope", np.zeros((3, 2)), np.ones(3))


def test_flatten_roundtrip():
    params = [np.arange(6.0).reshape(2, 3), np.array([7.0, 8.0])]
    back = unflatten_params(flatten_params(params), [p.shape for p in params])
    assert all(np.array_equal(a, b) for a, b in zip(params, back))


def test_attacks():
    assert flip_labels(np.array([0, 1, 1, 0]), 2).tolist() == [1, 0, 0, 1]
    g, l = [np.zeros(2)], [np.ones(2)]
    assert np.allclose(poison_update(g, l, 3.0)[0], -3.0)
    assert num_malicious(0.2, 10) == 2 and num_malicious(0.3, 10) == 3
