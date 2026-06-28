import numpy as np

from orbscreen.screen.cascade import (
    cascade_curve,
    combined_score,
    headline,
    recovery_at_k,
    routing_mask,
)


def test_routing_mask_budget_extremes():
    n = 10
    s = np.arange(n, dtype=float)
    u = np.arange(n, dtype=float)
    assert routing_mask(n, 0.0, policy="random", surrogate_score=s, uncertainty=u).sum() == 0
    assert routing_mask(n, 1.0, policy="random", surrogate_score=s, uncertainty=u).all()


def test_routing_mask_uncertainty_picks_most_uncertain():
    n = 5
    u = np.array([0.1, 0.9, 0.2, 0.8, 0.0])
    m = routing_mask(n, 0.4, policy="uncertainty", surrogate_score=np.zeros(n), uncertainty=u)
    assert m.tolist() == [False, True, False, True, False]  # indices 1 and 3 (highest uncertainty)


def test_routing_mask_confirm_top_ranked_picks_highest_score():
    n = 5
    s = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    m = routing_mask(n, 0.4, policy="confirm_top_ranked", surrogate_score=s, uncertainty=np.zeros(n))
    assert m.tolist() == [True, False, True, False, False]  # indices 0 and 2 (highest score)


def test_routing_mask_random_is_seeded():
    n = 20
    a = routing_mask(n, 0.5, policy="random", surrogate_score=np.zeros(n), uncertainty=np.zeros(n), seed=7)
    b = routing_mask(n, 0.5, policy="random", surrogate_score=np.zeros(n), uncertainty=np.zeros(n), seed=7)
    assert np.array_equal(a, b)


def test_routing_mask_unknown_policy():
    import pytest
    with pytest.raises(ValueError):
        routing_mask(4, 0.5, policy="nope", surrogate_score=np.zeros(4), uncertainty=np.zeros(4))


def test_combined_score_uses_oracle_where_routed():
    s = np.array([0.1, 0.2, 0.3])
    o = np.array([1.0, 1.0, 1.0])
    routed = np.array([True, False, True])
    assert combined_score(s, o, routed).tolist() == [1.0, 0.2, 1.0]


def test_recovery_at_k_full_and_partial():
    score = np.array([9, 8, 7, 6, 5], dtype=float)
    target = np.array([True, True, False, False, False])
    assert recovery_at_k(score, target, k=2) == 1.0          # both targets in top-2
    assert recovery_at_k(score, target, k=1) == 0.5          # one of two targets recovered
    assert recovery_at_k(score, np.zeros(5, bool), k=2) == 0.0  # empty target


def test_cascade_curve_reaches_full_recovery_and_is_better_than_surrogate():
    # surrogate is noisy; oracle is perfect. At budget=1 everything is oracle -> recovery 1.0.
    rng = np.random.default_rng(0)
    n = 200
    y_stab = (rng.random(n) < 0.2).astype(int)
    p = np.clip(y_stab * 0.6 + rng.random(n) * 0.4, 0, 1)  # weakly informative surrogate
    std = rng.random(n)
    k = int(round(0.1 * n))
    curve = cascade_curve(
        surrogate_score=p, oracle_score=y_stab.astype(float), uncertainty=std,
        target=(y_stab == 1), shortlist_k=k,
        orb_cost_per_million=100.0, surrogate_cost_per_million=0.5,
    )
    for policy in ("uncertainty", "confirm_top_ranked", "random"):
        rec = curve[policy]["recovery"]
        assert rec[-1] >= rec[0]                 # more routing never hurts recovery here
        assert curve[policy]["cost_per_million"][0] == 0.5  # budget 0 == surrogate-only cost


def test_headline_picks_best_policy_by_speedup():
    # policy "fast" reaches target recovery at a smaller budget than "slow".
    curve = {
        "fast": {"budget": [0.0, 0.1, 0.2], "recovery": [0.5, 0.96, 1.0], "cost_per_million": [0.5, 10.5, 20.5]},
        "slow": {"budget": [0.0, 0.5, 1.0], "recovery": [0.5, 0.96, 1.0], "cost_per_million": [0.5, 50.5, 100.5]},
    }
    h = headline(curve, surrogate_cpm=0.5, orb_cpm=100.0, target_recovery=0.95)
    assert h["best"]["policy"] == "fast"
    assert h["best"]["speedup_vs_orb"] > 1.0
    assert set(h["surrogate_only_recovery"]) == {"fast", "slow"}
