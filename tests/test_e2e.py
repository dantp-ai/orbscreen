"""Offline end-to-end pipeline test: build_dataset -> train_baseline (no network)."""

import numpy as np

from orbscreen.data.build import build_dataset
from orbscreen.models.baseline import train_baseline


def test_end_to_end_build_and_baseline(fake_dbs_large, tmp_path):
    out = tmp_path / "ds.parquet"
    df, stats = build_dataset(fake_dbs_large["samples"], fake_dbs_large["relaxed"], out)
    assert out.exists()
    assert stats["paired"] == 100 and stats["formula_mismatch"] == 0

    # Force a non-degenerate split: interleave by stability so train and test each
    # contain both classes (keeps the classifier + AUROC well-defined on small data).
    order = df.sort_values("stability").index.to_numpy()
    df = df.copy()
    df.loc[order[0::2], "split_random"] = "train"
    df.loc[order[1::2], "split_random"] = "test"

    res = train_baseline(df, split_col="split_random")
    assert set(res.keys()) == {"regression", "classification", "ranking"}
    assert np.isfinite(res["regression"]["mae"])
    assert 0.0 <= res["classification"]["auroc"] <= 1.0
