# src/orbscreen/screen/report.py
"""Tie screen artifacts together into the cascade results: cost model + Pareto + headline.

Inputs are the artifacts produced by the Modal `screen` and `benchmark_orb` runs:
- screen_predictions.parquet : per-structure predictions + ground-truth labels
- screen_timing.json         : {"throughput_per_sec": surrogate inference throughput}
- benchmark_orb.json         : {"throughput_per_sec": measured Orb-v3 relaxation throughput}
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from orbscreen.screen.cascade import cascade_curve, headline
from orbscreen.screen.cost import dollars_per_million, end_to_end_throughput
from orbscreen.screen.plots import plot_cascade_pareto


def _energy_target(y_energy, k):
    """Boolean mask of the k lowest-energy (most stable) structures = Orb-v3's top-k by energy."""
    target = np.zeros(len(y_energy), dtype=bool)
    target[np.argsort(y_energy)[:k]] = True
    return target


def run_cascade_analysis(predictions_path, screen_timing_path, orb_benchmark_path, out_dir, *,
                         shortlist_frac=0.10, target_recovery=0.95, usd_per_hour=None):
    df = pd.read_parquet(predictions_path)
    n = len(df)
    k = max(1, int(round(shortlist_frac * n)))
    surrogate_tp = json.loads(Path(screen_timing_path).read_text())["throughput_per_sec"]
    orb_tp = json.loads(Path(orb_benchmark_path).read_text())["throughput_per_sec"]
    rate = {} if usd_per_hour is None else {"usd_per_hour": usd_per_hour}
    surrogate_cpm = dollars_per_million(surrogate_tp, **rate)
    surrogate_cpm_e2e = dollars_per_million(end_to_end_throughput(surrogate_tp), **rate)
    orb_cpm = dollars_per_million(orb_tp, **rate)

    rankings = {
        "stability": dict(
            surrogate_score=df["p_stable"].to_numpy(dtype=float),
            oracle_score=df["y_stab"].to_numpy(dtype=float),
            uncertainty=df["p_stable_std"].to_numpy(dtype=float),
            target=(df["y_stab"].to_numpy() == 1)),
        "energy": dict(
            surrogate_score=-df["energy_pred"].to_numpy(dtype=float),
            oracle_score=-df["y_energy"].to_numpy(dtype=float),
            uncertainty=df["energy_std"].to_numpy(dtype=float),
            target=_energy_target(df["y_energy"].to_numpy(dtype=float), k)),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cascades, headlines = {}, {}
    for name, r in rankings.items():
        curve = cascade_curve(
            surrogate_score=r["surrogate_score"], oracle_score=r["oracle_score"],
            uncertainty=r["uncertainty"], target=r["target"], shortlist_k=k,
            orb_cost_per_million=orb_cpm, surrogate_cost_per_million=surrogate_cpm)
        cascades[name] = curve
        headlines[name] = headline(curve, surrogate_cpm=surrogate_cpm, orb_cpm=orb_cpm,
                                   target_recovery=target_recovery)
        plot_cascade_pareto(curve, ranking_name=name, out_path=str(out / f"cascade_{name}.png"))

    results = {
        "n": n, "shortlist_k": k, "shortlist_frac": shortlist_frac,
        "surrogate_throughput_per_sec": surrogate_tp, "orb_throughput_per_sec": orb_tp,
        "surrogate_cost_per_million": surrogate_cpm,
        "surrogate_cost_per_million_end_to_end": surrogate_cpm_e2e,
        "orb_cost_per_million": orb_cpm,
        "headline": headlines,
    }
    (out / "results_screen.json").write_text(json.dumps(results, indent=2))
    (out / "cascade.json").write_text(json.dumps(cascades, indent=2))
    return results
