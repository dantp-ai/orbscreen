"""Evaluation metrics for OrbScreen models.

All functions accept array-likes; numpy arrays are returned or plain Python floats.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, and Spearman correlation."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "spearman": float(rho),
    }


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """AUROC, AUPRC (average precision), and F1 at *threshold*.

    AUPRC is the threshold-free metric to trust under class imbalance; F1 at a fixed
    0.5 threshold is reported for reference but is misleading when positives are rare.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, (y_prob >= threshold).astype(int))),
    }


def precision_at_k(y_true, score, k: int) -> float:
    """Fraction of positives among the top-*k* ranked candidates.

    Ties are broken by ``np.argsort`` order (stable, input-order dependent); results are
    deterministic for a fixed input order.
    """
    y_true, score = np.asarray(y_true, dtype=int), np.asarray(score, dtype=float)
    top = np.argsort(score)[::-1][:k]
    return float(y_true[top].mean())


def enrichment_factor(y_true, score, k: int) -> float:
    """Precision@k divided by the overall positive rate.

    Returns 0.0 when the base rate is zero to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=int)
    base = float(y_true.mean())
    if base == 0.0:
        return 0.0
    return precision_at_k(y_true, score, k) / base


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Weighted average gap between confidence and accuracy across probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(conf - acc)
    return float(ece)
