"""Assemble the unified Phase 1 dataset from the two MofasaDB ASE DBs.

Pairing is POSITIONAL by ASE ``row.id`` (structure_id/mofid are not unique keys; see
docs/data-schema.md), guarded by chemical-formula equality (relaxation preserves atoms).

Phase 1 stores tabular features + targets only (no serialised structures); the GNN phase
adds a structure export step.
"""

from pathlib import Path

import pandas as pd
from ase.db import connect

from orbscreen.data.parse import relaxed_fields, sample_fields
from orbscreen.data.splits import random_split, topology_split
from orbscreen.data.targets import stability_label


def build_records(unrelaxed_path, relaxed_path, limit: int | None = None) -> tuple[list[dict], dict]:
    """Pair rows positionally and return (records, stats).

    Each record is flat: id, formula, n_atoms, topology, geom_* features, validity flags,
    energy_per_atom (target), geo_converged, relaxed_max_force, orb_energy_unrelaxed, stability.
    """
    relaxed = {row.id: (row.formula, relaxed_fields(row)) for row in connect(str(relaxed_path)).select()}
    records: list[dict] = []
    mismatch = 0
    missing = 0
    for row in connect(str(unrelaxed_path)).select():
        pair = relaxed.get(row.id)
        if pair is None:
            missing += 1
            continue
        rformula, rfields = pair
        if row.formula != rformula:
            mismatch += 1
            continue
        sf = sample_fields(row)
        geom = sf.pop("geometry")
        rec = {"id": int(row.id), **sf, **{f"geom_{k}": v for k, v in geom.items()}, **rfields}
        rec["stability"] = stability_label(rec)
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    stats = {"paired": len(records), "formula_mismatch": mismatch, "missing_relaxed": missing}
    return records, stats


def build_dataset(unrelaxed_path, relaxed_path, out_path, limit: int | None = None, seed: int = 0):
    """Build the dataset, add both splits, write Parquet; return (DataFrame, stats)."""
    records, stats = build_records(unrelaxed_path, relaxed_path, limit=limit)
    df = pd.DataFrame.from_records(records)
    df["split_random"] = random_split(df, seed=seed).values
    df["split_topology"] = topology_split(df, seed=seed).values
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df, stats
