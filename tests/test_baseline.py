"""Tests for orbscreen.models.baseline."""

import numpy as np
import pandas as pd

from orbscreen.models.baseline import train_baseline


def _synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic dataset matching the real dataset schema.

    energy_per_atom is a deterministic linear function of pld and asa_m2_per_g
    plus small noise, so the regressor has a learnable signal. stability is a
    SEPARATE feature-learnable signal (large pores), deliberately NOT identical to
    the energy ordering — this mirrors the real flag-based label and ensures the
    ranking metric is exercised against P(stable), not energy.
    """
    rng = np.random.default_rng(seed)
    pld = rng.uniform(2, 12, n)
    lcd = pld + rng.uniform(0.5, 2.0, n)
    asa = rng.uniform(500, 4000, n)
    epa = -8.0 + 0.1 * pld - 0.0005 * asa + rng.normal(0, 0.05, n)
    stability = ((lcd > 8.0) & (asa > 2000)).astype(int)

    # assign splits roughly 80/10/10
    split_vals = rng.choice(
        ["train", "val", "test"], n, p=[0.8, 0.1, 0.1]
    )

    return pd.DataFrame(
        {
            "formula": rng.choice(["Zn4O4", "Cu2O2", "Mg2C2O4", "Al4O6"], n),
            "n_atoms": rng.integers(6, 40, n),
            "geom_pld": pld,
            "geom_lcd": lcd,
            "geom_asa_m2_per_g": asa,
            "energy_per_atom": epa,
            "stability": stability,
            "split_random": split_vals,
        }
    )


def test_baseline_learns_regression_signal():
    df = _synthetic()
    res = train_baseline(df, split_col="split_random")
    # A model that just predicts the mean would get ~0.2 MAE on this data.
    assert res["regression"]["mae"] < 0.2


def test_baseline_learns_classification_signal():
    df = _synthetic()
    res = train_baseline(df, split_col="split_random")
    assert res["classification"]["auroc"] > 0.7


def test_baseline_returns_all_keys():
    df = _synthetic(n=200)
    res = train_baseline(df, split_col="split_random")
    assert set(res.keys()) == {"regression", "classification", "ranking"}
    assert set(res["regression"].keys()) == {"mae", "rmse", "spearman"}
    assert set(res["classification"].keys()) == {"auroc", "auprc", "f1"}
    assert set(res["ranking"].keys()) == {"base_rate", "precision_at_10pct", "enrichment_at_10pct"}


def test_baseline_ranking_enriches_for_stability():
    # ranking by P(stable) must beat random (enrichment > 1) on a learnable signal
    df = _synthetic(n=800)
    res = train_baseline(df, split_col="split_random")
    assert res["ranking"]["enrichment_at_10pct"] > 1.0
