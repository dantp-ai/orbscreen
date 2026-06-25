"""Tests for orbscreen.features.composition."""

from orbscreen.features.composition import composition_features


def test_composition_feature_keys():
    f = composition_features("Zn4C8H4O8")
    assert "mean_X" in f
    assert "std_atomic_mass" in f
    assert f["n_elements"] == 4


def test_composition_feature_mean_x_positive():
    f = composition_features("Zn4C8H4O8")
    assert f["mean_X"] > 0


def test_composition_single_element():
    f = composition_features("Fe8")
    assert f["n_elements"] == 1
    assert f["std_atomic_mass"] == 0.0  # no variance with one element


def test_composition_cached_consistent():
    """Calling twice (cache hit) returns the same object."""
    f1 = composition_features("Cu2O")
    f2 = composition_features("Cu2O")
    assert f1 is f2  # lru_cache returns same dict instance
