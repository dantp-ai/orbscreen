import numpy as np
import pandas as pd
import pytest

gr = pytest.importorskip("gradio")  # skipped in CI (app extra not installed)

from orbscreen.app.demo import build_demo  # noqa: E402


def _preds():
    n = 20
    return pd.DataFrame({
        "gid": list(range(n)),
        "p_stable": np.linspace(0.1, 0.95, n),
        "p_stable_std": np.full(n, 0.02),
        "energy_pred": np.linspace(-6.5, -6.0, n),
        "energy_std": np.full(n, 0.03),
        "geom_pld": np.linspace(2.0, 8.0, n),
        "geom_asa_m2_per_g": np.linspace(0.0, 3000.0, n),
    })


def test_build_demo_returns_blocks():
    demo = build_demo(_preds(), checkpoints=["dummy.pt"])
    assert isinstance(demo, gr.Blocks)
