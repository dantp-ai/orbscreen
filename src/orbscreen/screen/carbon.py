"""Carbon-capture screening score: a geometric proxy combining predicted stability with
pyzeo pore geometry. NOT GCMC - an explicit shortlist heuristic.

score = P(stable) * 1[PLD >= 3.3 A] * normalize(gravimetric accessible surface area)
The 3.3 A pore-limiting-diameter gate is CO2's kinetic diameter (the sensitivity knob).
"""

import numpy as np
import pandas as pd

PLD_MIN = 3.3  # CO2 kinetic diameter (Angstrom)


def carbon_capture_score(p_stable: float, pld: float, asa_normalized: float) -> float:
    """Single-MOF score: stability x CO2-accessibility gate x normalized surface area."""
    gate = 1.0 if pld >= PLD_MIN else 0.0
    return float(p_stable) * gate * float(asa_normalized)


def _normalize_asa(asa, clip_pct: float = 99.0) -> np.ndarray:
    """Clipped min-max of gravimetric surface area to [0, 1] over valid entries; NaN -> 0."""
    arr = np.asarray(asa, dtype=float)
    valid = ~np.isnan(arr)
    result = np.zeros(len(arr))
    if valid.any():
        vals = arr[valid]
        lo = float(vals.min())
        hi = float(np.percentile(vals, clip_pct))
        if hi > lo:
            result[valid] = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
    return result


def score_corpus(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the predictions with a `carbon_score` column (corpus-normalized)."""
    df = predictions_df.copy()
    asa_norm = _normalize_asa(df["geom_asa_m2_per_g"].to_numpy())
    pld = np.nan_to_num(df["geom_pld"].to_numpy(dtype=float), nan=-1.0)
    gate = (pld >= PLD_MIN).astype(float)
    df["carbon_score"] = df["p_stable"].to_numpy(dtype=float) * gate * asa_norm
    return df


def rank_corpus(predictions_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Top-N carbon-capture leaderboard with score + key descriptors."""
    scored = score_corpus(predictions_df)
    cols = ["gid", "carbon_score", "p_stable", "geom_pld", "geom_asa_m2_per_g", "energy_pred"]
    return (
        scored.sort_values("carbon_score", ascending=False)
        .head(top_n)[cols]
        .reset_index(drop=True)
    )
