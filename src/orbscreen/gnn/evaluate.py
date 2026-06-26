"""Deep-ensemble evaluation on a split's test set: accuracy, ranking, calibration."""

import torch
from torch_geometric.loader import DataLoader

from orbscreen.eval.metrics import (
    classification_metrics,
    enrichment_factor,
    expected_calibration_error,
    precision_at_k,
    regression_metrics,
)
from orbscreen.gnn.dataset import reslice_split
from orbscreen.gnn.ensemble import ensemble_predict


def evaluate_ensemble(checkpoints, samples_db, parquet, split, cache_dir=".graph_cache") -> dict:
    """Evaluate an ensemble of checkpoints on the test set of `split`.

    The test set is sliced from the already-built graph caches (no rebuild for any split).
    Reports energy regression, stability classification, ranking (enrichment@10%), and
    uncertainty (ECE + mean predictive std from the ensemble).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_graphs = reslice_split(samples_db, parquet, cache_dir, split, "test")
    pred = ensemble_predict(checkpoints, DataLoader(test_graphs, batch_size=64), device)
    y, p = pred["y_stab"], pred["stab_mean"]
    k = max(1, int(0.1 * len(y)))
    return {
        "split": split,
        "n_models": len(checkpoints),
        "n_test": int(len(y)),
        "regression": regression_metrics(pred["y_energy"], pred["energy_mean"]),
        "classification": classification_metrics(y, p),
        "ranking": {
            "base_rate": float(y.mean()),
            "precision_at_10pct": precision_at_k(y, p, k),
            "enrichment_at_10pct": enrichment_factor(y, p, k),
        },
        "uncertainty": {
            "ece": expected_calibration_error(y, p),
            "mean_energy_std": float(pred["energy_std"].mean()),
            "mean_stab_std": float(pred["stab_std"].mean()),
        },
    }
