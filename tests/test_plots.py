# tests/test_plots.py
from orbscreen.screen.plots import plot_cascade_pareto


def test_plot_cascade_pareto_writes_file(tmp_path):
    curve = {
        "uncertainty": {"budget": [0.0, 0.5, 1.0], "cost_per_million": [0.5, 50.0, 100.0],
                        "recovery": [0.6, 0.9, 1.0]},
        "random": {"budget": [0.0, 0.5, 1.0], "cost_per_million": [0.5, 50.0, 100.0],
                   "recovery": [0.6, 0.75, 1.0]},
    }
    out = tmp_path / "cascade_stability.png"
    result = plot_cascade_pareto(curve, ranking_name="stability", out_path=str(out))
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0
