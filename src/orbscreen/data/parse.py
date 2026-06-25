"""Extract fields from MofasaDB ASE rows.

All MofasaDB content lives in ``row.data`` (``key_value_pairs`` is empty), with
energies/forces and geometry nested under ``row.data['properties']`` and validity
flags under ``row.data['metrics']``. See docs/data-schema.md.
"""

import math

from orbscreen import config


def to_float(v) -> float:
    """Coerce a MofasaDB value to float; 'None'/None/non-numeric -> NaN.

    Geometry values are stored as strings (e.g. '2.61') or the string 'None';
    some energies are stored as single-element lists.
    """
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) == 1 else math.nan
    if v is None:
        return math.nan
    if isinstance(v, str) and v.strip().lower() in {"none", "nan", ""}:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def nested(d, *path, default=None):
    """Safe nested dict lookup: nested(d, 'properties', 'orb_properties', ...)."""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def as_dict(v) -> dict:
    """Return v if it is a dict, else {}.

    MofasaDB stores failed sub-records as the string 'None' (e.g. pyzeo geometry or
    mofchecker when those computations failed), so a dict is not guaranteed.
    """
    return v if isinstance(v, dict) else {}


def sample_fields(row) -> dict:
    """Fields from an unrelaxed (samples.db) row: geometry features + validity flags.

    The unrelaxed Orb energy is captured as a reference only (Orb-dependent, so not
    used as a surrogate input).
    """
    data = row.data
    props = as_dict(data.get("properties"))
    metrics = as_dict(data.get("metrics"))
    geom_raw = as_dict(nested(props, "pyzeo_geometric_properties"))
    mofchecker = as_dict(nested(metrics, "mofchecker"))
    return {
        "topology": data.get("topology"),
        "formula": row.formula,
        "n_atoms": int(row.natoms),
        "geometry": {k: to_float(geom_raw.get(k)) for k in config.GEOMETRY_KEYS},
        "orb_energy_unrelaxed": to_float(nested(props, "orb_properties", "orb_energy_per_atom")),
        "smact_valid": bool(metrics.get("smact_valid")),
        "no_atom_too_close": bool(metrics.get("no_atom_too_close")),
        "reconstruction_failed": bool(metrics.get("reconstruction_failed")),
        "mofchecker_valid": bool(mofchecker.get("mofchecker_valid")),
    }


def relaxed_fields(row) -> dict:
    """Fields from a relaxed (relaxed.db) row: the Orb-v3 relaxed-state target."""
    props = as_dict(row.data.get("properties"))
    return {
        "energy_per_atom": to_float(nested(props, "orb_properties", "orb_energy_per_atom")),
        "relaxed_max_force": to_float(nested(props, "orb_properties", "orb_max_force")),
        "geo_converged": bool(row.data.get("geo_converged")),
    }
