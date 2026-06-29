"""Cost-accuracy cascade: route hard candidates to the Orb-v3 oracle and measure how
much of Orb-v3's top-k the cascade recovers as a function of routing budget.

Pure analysis over a predictions table; no GPU, no Orb-v3 calls. The oracle "score" is the
ground-truth label already in the predictions table; the cost of routing is added separately.
"""

import numpy as np

from orbscreen.screen.cost import cascade_cost_per_million, speedup

POLICIES = ("uncertainty", "confirm_top_ranked", "random")


def routing_mask(
    n: int,
    budget: float,
    *,
    policy: str,
    surrogate_score: np.ndarray,
    uncertainty: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Boolean mask of which of `n` candidates are routed to Orb-v3 at `budget` in [0, 1].

    Policies: 'uncertainty' (most-uncertain first), 'confirm_top_ranked' (highest
    surrogate score first), 'random' (seeded).
    """
    n_route = int(round(budget * n))
    mask = np.zeros(n, dtype=bool)
    if n_route <= 0:
        return mask
    if policy == "uncertainty":
        order = np.argsort(uncertainty)[::-1]
    elif policy == "confirm_top_ranked":
        order = np.argsort(surrogate_score)[::-1]
    elif policy == "random":
        order = np.random.default_rng(seed).permutation(n)
    else:
        raise ValueError(f"unknown routing policy: {policy!r}")
    mask[order[:n_route]] = True
    return mask


def combined_score(
    surrogate_score: np.ndarray, oracle_score: np.ndarray, routed: np.ndarray
) -> np.ndarray:
    """Oracle score where routed, surrogate score otherwise."""
    return np.where(routed, oracle_score, surrogate_score)


def recovery_at_k(score: np.ndarray, target: np.ndarray, k: int) -> float:
    """Recall of the boolean `target` set within the top-`k` by `score`: |top_k & target| / |target|."""
    target = np.asarray(target, dtype=bool)
    n_target = int(target.sum())
    if n_target == 0 or k <= 0:
        return 0.0
    top = np.argsort(score)[::-1][:k]
    return float(target[top].sum() / n_target)


def cascade_curve(
    *,
    surrogate_score: np.ndarray,
    oracle_score: np.ndarray,
    uncertainty: np.ndarray,
    target: np.ndarray,
    shortlist_k: int,
    orb_cost_per_million: float,
    surrogate_cost_per_million: float,
    policies: tuple = POLICIES,
    budgets: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """Recovery-vs-cost points per routing policy.

    Returns {policy: {"budget": [...], "cost_per_million": [...], "recovery": [...]}}.
    """
    surrogate_score = np.asarray(surrogate_score, dtype=float)
    oracle_score = np.asarray(oracle_score, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    n = len(surrogate_score)
    if budgets is None:
        budgets = np.linspace(0.0, 1.0, 21)
    out = {}
    for policy in policies:
        rec, cost = [], []
        for b in budgets:
            routed = routing_mask(
                n,
                b,
                policy=policy,
                surrogate_score=surrogate_score,
                uncertainty=uncertainty,
                seed=seed,
            )
            cs = combined_score(surrogate_score, oracle_score, routed)
            rec.append(recovery_at_k(cs, target, shortlist_k))
            cost.append(
                cascade_cost_per_million(
                    float(b), surrogate_cost_per_million, orb_cost_per_million
                )
            )
        out[policy] = {
            "budget": [float(b) for b in budgets],
            "cost_per_million": cost,
            "recovery": rec,
        }
    return out


def headline(
    curve: dict, *, surrogate_cpm: float, orb_cpm: float, target_recovery: float = 0.95
) -> dict:
    """Per policy, the cheapest budget reaching `target_recovery` and its speedup vs full Orb-v3.

    Returns {"target_recovery", "best" (highest-speedup policy or None), "per_policy" (sorted),
    "surrogate_only_recovery" (recovery at budget 0 per policy)}.
    """
    rows = []
    for policy, d in curve.items():
        hits = [
            (b, r) for b, r in zip(d["budget"], d["recovery"]) if r >= target_recovery
        ]
        if not hits:
            continue
        b, r = min(hits, key=lambda x: x[0])
        cpm = cascade_cost_per_million(b, surrogate_cpm, orb_cpm)
        rows.append(
            {
                "policy": policy,
                "budget": b,
                "recovery": r,
                "cost_per_million": cpm,
                "speedup_vs_orb": speedup(orb_cpm, cpm),
            }
        )
    rows.sort(key=lambda x: x["speedup_vs_orb"], reverse=True)
    return {
        "target_recovery": target_recovery,
        "best": rows[0] if rows else None,
        "per_policy": rows,
        "surrogate_only_recovery": {p: d["recovery"][0] for p, d in curve.items()},
    }
