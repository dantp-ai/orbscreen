import numpy as np
import pandas as pd

from orbscreen.screen.carbon import (
    PLD_MIN,
    carbon_capture_score,
    rank_corpus,
    score_corpus,
)


def test_pld_gate_blocks_below_threshold():
    assert PLD_MIN == 3.3
    assert carbon_capture_score(0.9, pld=3.0, asa_normalized=1.0) == 0.0
    assert carbon_capture_score(0.9, pld=3.3, asa_normalized=1.0) == 0.9


def test_score_is_product_of_three_factors():
    assert carbon_capture_score(0.5, pld=5.0, asa_normalized=0.4) == 0.2


def _df():
    return pd.DataFrame({
        "gid": [1, 2, 3, 4],
        "p_stable": [0.9, 0.9, 0.2, 0.95],
        "geom_pld": [5.0, 2.0, 6.0, 4.0],          # gid=2 fails the CO2 gate
        # gid=1 and gid=3 share ASA (2000); gid=2 lower so valid range isn't degenerate;
        # gid=4 has missing ASA.
        "geom_asa_m2_per_g": [2000.0, 800.0, 2000.0, np.nan],
        "energy_pred": [-6.0, -6.1, -6.2, -6.3],
    })


def test_score_corpus_gates_and_handles_nan():
    scored = score_corpus(_df())
    s = scored.set_index("gid")["carbon_score"]
    assert s[2] == 0.0          # PLD below 3.3 -> gated out
    assert s[4] == 0.0          # NaN surface area -> treated as 0
    assert s[1] > s[3]          # same ASA, higher P(stable) ranks higher
    assert (scored["carbon_score"] >= 0).all()


def test_normalization_ignores_nan_for_baseline():
    df = pd.DataFrame({
        "gid": [1, 2, 3],
        "p_stable": [1.0, 1.0, 1.0],
        "geom_pld": [5.0, 5.0, 5.0],
        "geom_asa_m2_per_g": [1000.0, 2000.0, np.nan],
        "energy_pred": [-6.0, -6.0, -6.0],
    })
    scored = score_corpus(df).set_index("gid")
    assert scored.loc[1, "carbon_score"] == 0.0   # lowest VALID asa -> normalizes to 0
    assert scored.loc[3, "carbon_score"] == 0.0   # NaN asa -> 0
    assert scored.loc[2, "carbon_score"] > 0.0


def test_rank_corpus_orders_and_truncates():
    out = rank_corpus(_df(), top_n=2)
    assert list(out.columns) == [
        "gid",
        "carbon_score",
        "p_stable",
        "geom_pld",
        "geom_asa_m2_per_g",
        "energy_pred",
    ]
    assert len(out) == 2
    assert out["carbon_score"].is_monotonic_decreasing
    assert out.iloc[0]["gid"] == 1   # highest score first
