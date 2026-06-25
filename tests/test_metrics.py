"""Tests for orbscreen.eval.metrics."""

import numpy as np

from orbscreen.eval.metrics import (
    classification_metrics,
    enrichment_factor,
    precision_at_k,
    regression_metrics,
)


def test_regression_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["spearman"] > 0.99


def test_regression_metrics_returns_all_keys():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y + 0.1)
    assert set(m.keys()) == {"mae", "rmse", "spearman"}


def test_classification_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y, p)
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
    assert m["f1"] == 1.0


def test_classification_metrics_returns_all_keys():
    y = np.array([0, 1])
    p = np.array([0.3, 0.7])
    m = classification_metrics(y, p)
    assert set(m.keys()) == {"auroc", "auprc", "f1"}


def test_precision_at_k_perfect():
    # first two candidates are positives, score ranks them highest
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.arange(10)[::-1]  # score[0]=9, score[1]=8, ...
    assert precision_at_k(y, score, 2) == 1.0


def test_precision_at_k_partial():
    y = np.array([1, 0, 0, 0])
    score = np.array([3, 2, 1, 0])  # top-2 are index 0 (positive) and 1 (negative)
    assert precision_at_k(y, score, 2) == 0.5


def test_enrichment_factor():
    # base rate = 2/10 = 0.2, precision@2 = 1.0, EF = 5.0
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.arange(10)[::-1]
    assert enrichment_factor(y, score, 2) == 5.0


def test_enrichment_factor_zero_base_rate():
    y = np.zeros(10, dtype=int)
    score = np.arange(10, dtype=float)
    assert enrichment_factor(y, score, 3) == 0.0
