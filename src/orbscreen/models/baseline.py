"""Descriptor-based gradient-boosted-tree baseline for Phase 1.

Uses ``HistGradientBoostingRegressor`` on ``energy_per_atom`` and
``HistGradientBoostingClassifier`` on ``stability``.  The screening score for the
stability objective is the classifier's predicted ``P(stable)`` (the ``stability``
label is validity-flag-based, so ranking by energy would not recover it).
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from orbscreen.eval.metrics import (
    classification_metrics,
    enrichment_factor,
    precision_at_k,
    regression_metrics,
)
from orbscreen.features.descriptors import build_feature_matrix


def train_baseline(df: pd.DataFrame, split_col: str = "split_random") -> dict:
    """Fit regressor + classifier on train split, evaluate on test split.

    Args:
        df: The unified dataset DataFrame produced by ``build_dataset``.
        split_col: Column name holding the split labels (``"split_random"`` or
            ``"split_topology"``).

    Returns:
        Dict with keys ``"regression"``, ``"classification"``, ``"ranking"``.
        Ranking precision and enrichment are measured at 10 % of the test set.
    """
    # Features may contain NaN (missing geometry); HistGradientBoosting handles NaN
    # natively, so no imputation is needed here. The "val" split is reserved for
    # Phase 2 (early stopping / HP tuning); Phase 1 evaluates on train/test only.
    X, _ = build_feature_matrix(df)
    train = df[split_col] == "train"
    test = df[split_col] == "test"

    reg = HistGradientBoostingRegressor(random_state=0).fit(
        X[train], df.loc[train, "energy_per_atom"]
    )
    clf = HistGradientBoostingClassifier(random_state=0).fit(
        X[train], df.loc[train, "stability"]
    )

    epa_pred = reg.predict(X[test])
    prob = clf.predict_proba(X[test])[:, 1]
    y_epa = df.loc[test, "energy_per_atom"].to_numpy()
    y_stab = df.loc[test, "stability"].to_numpy()
    score = prob  # screening by predicted P(stable) — the stability objective

    k = max(1, int(0.1 * len(y_stab)))
    return {
        "regression": regression_metrics(y_epa, epa_pred),
        "classification": classification_metrics(y_stab, prob),
        "ranking": {
            "base_rate": float(y_stab.mean()),
            "precision_at_10pct": precision_at_k(y_stab, score, k),
            "enrichment_at_10pct": enrichment_factor(y_stab, score, k),
        },
    }
