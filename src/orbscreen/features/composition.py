"""Fraction-weighted elemental composition features from a chemical formula string."""

import functools

import numpy as np
from pymatgen.core import Composition, Element

_PROPS: dict[str, object] = {
    "atomic_mass": lambda e: float(e.atomic_mass),
    "X": lambda e: float(e.X) if e.X else np.nan,
    "atomic_radius": lambda e: float(e.atomic_radius) if e.atomic_radius else np.nan,
    "row": lambda e: float(e.row),
    "group": lambda e: float(e.group),
}


@functools.lru_cache(maxsize=None)
def composition_features(formula: str) -> dict[str, float]:
    """Return fraction-weighted elemental statistics for *formula*.

    Keys: ``n_elements``, ``mean_<prop>``, ``std_<prop>`` for each property in
    ``_PROPS``.  Missing element data is skipped gracefully (set to 0.0).
    """
    comp = Composition(formula)
    fracs = comp.fractional_composition.get_el_amt_dict()
    elems: list[tuple[Element, float]] = [(Element(sym), w) for sym, w in fracs.items()]
    out: dict[str, float] = {"n_elements": float(len(elems))}
    for name, fn in _PROPS.items():
        vals = np.array([fn(e) for e, _ in elems], dtype=float)  # type: ignore[operator]
        weights = np.array([w for _, w in elems], dtype=float)
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            out[f"mean_{name}"] = 0.0
            out[f"std_{name}"] = 0.0
            continue
        v, w = vals[mask], weights[mask]
        mean = float((v * w).sum() / w.sum())
        out[f"mean_{name}"] = mean
        out[f"std_{name}"] = float(np.sqrt((w * (v - mean) ** 2).sum() / w.sum()))
    return out
