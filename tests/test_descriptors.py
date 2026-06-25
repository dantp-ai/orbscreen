"""Tests for orbscreen.features.descriptors."""

import math

import pandas as pd

from orbscreen.features.descriptors import build_feature_matrix


def _small_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "formula": ["Zn4O4", "Cu2C4H2O4"],
            "n_atoms": [8, 12],
            "geom_pld": [4.0, 5.0],
            "geom_lcd": [6.0, None],  # intentional NaN
            "geom_asa_m2_per_g": [1500.0, 1800.0],
            # columns that must NOT appear in features:
            "orb_energy_unrelaxed": [-6.0, -6.5],
            "energy_per_atom": [-7.0, -7.5],
            "stability": [1, 0],
            "split_random": ["train", "test"],
        }
    )


def test_feature_matrix_shape():
    df = _small_df()
    X, names = build_feature_matrix(df)
    assert len(X) == 2
    assert len(names) == X.shape[1]


def test_feature_matrix_includes_geom_and_composition():
    df = _small_df()
    X, names = build_feature_matrix(df)
    assert "geom_pld" in names
    assert "geom_lcd" in names
    assert any(n.startswith("mean_") for n in names)
    assert "n_atoms" in names


def test_feature_matrix_preserves_nan_for_model():
    """Missing geometry is left as NaN (no leaky whole-dataset imputation)."""
    df = _small_df()
    X, _ = build_feature_matrix(df)
    assert math.isnan(X["geom_lcd"].iloc[1])  # the intentional NaN is preserved


def test_feature_matrix_excludes_targets_and_flags():
    df = _small_df()
    _, names = build_feature_matrix(df)
    forbidden = {"orb_energy_unrelaxed", "energy_per_atom", "stability", "split_random"}
    assert not forbidden & set(names)
