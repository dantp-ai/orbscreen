# src/orbscreen/screen/plots.py
"""Plots for the Phase 3 cascade (recovery vs compute cost)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_cascade_pareto(curve, *, ranking_name, out_path):
    """Recovery vs $/million for each routing policy; saves a PNG and returns its path."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for policy, d in curve.items():
        ax.plot(d["cost_per_million"], d["recovery"], marker="o", label=policy)
    ax.set_xscale("log")
    ax.set_xlabel("compute cost ($ / million structures)")
    ax.set_ylabel(f"recovery of Orb-v3 top-k ({ranking_name})")
    ax.set_title(f"Cost-accuracy cascade - {ranking_name} ranking")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
