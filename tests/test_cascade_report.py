# tests/test_cascade_report.py
import json

import numpy as np
import pandas as pd

from orbscreen.screen.report import run_cascade_analysis


def _make_inputs(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    y_stab = (rng.random(n) < 0.2).astype(int)
    y_energy = rng.normal(-6.5, 0.2, n)
    df = pd.DataFrame({
        "gid": np.arange(n),
        "p_stable": np.clip(y_stab * 0.6 + rng.random(n) * 0.4, 0, 1),
        "p_stable_std": rng.random(n) * 0.1,
        "energy_pred": y_energy + rng.normal(0, 0.05, n),
        "energy_std": rng.random(n) * 0.05,
        "y_stab": y_stab,
        "y_energy": y_energy,
    })
    pred = tmp_path / "screen_predictions.parquet"
    df.to_parquet(pred, index=False)
    (tmp_path / "screen_timing.json").write_text(json.dumps({"throughput_per_sec": 5000.0}))
    (tmp_path / "benchmark_orb.json").write_text(json.dumps({"throughput_per_sec": 0.2}))
    return pred


def test_run_cascade_analysis_writes_artifacts(tmp_path):
    pred = _make_inputs(tmp_path)
    res = run_cascade_analysis(
        str(pred), str(tmp_path / "screen_timing.json"), str(tmp_path / "benchmark_orb.json"),
        str(tmp_path), usd_per_hour=1.10,
    )
    for name in ("results_screen.json", "cascade.json", "cascade_stability.png", "cascade_energy.png"):
        assert (tmp_path / name).exists()
    assert res["orb_cost_per_million"] > res["surrogate_cost_per_million"]
    assert set(res["headline"]) == {"stability", "energy"}
    # the cascade is cheaper than full Orb-v3 at the target recovery
    best = res["headline"]["stability"]["best"]
    assert best is None or best["speedup_vs_orb"] > 1.0
