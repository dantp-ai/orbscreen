# MofasaDB — confirmed schema (from real data, 2026-06-26)

Source: `Orbital-Materials/MofasaDB` (CC-BY-4.0). Two ASE SQLite DBs we use:
- `samples.db` — 201,926 rows, **unrelaxed** generated MOFs (2.26 GB)
- `relaxed.db` — 201,926 rows, **Orb-v3-relaxed** MOFs (2.19 GB)
- (`*_latents/*.npy` ≈ 14 GB of embeddings — NOT needed for the descriptor baseline)

## Critical: pairing is POSITIONAL by ASE `row.id`

`structure_id` and `mofid` are **NOT** usable join keys:
- `structure_id` is a within-shard counter (repeats every ~1592 rows) and is numbered
  differently in `samples.db` vs `relaxed.db` (e.g. global row id 50000 → samples
  `structure_id=2084`, relaxed `structure_id=648`).
- `mofid` differs across the two DBs for the same structure (samples tags `.ERROR.`,
  relaxed tags `.UNKNOWN.`).

**Pair `samples.db` row N with `relaxed.db` row N (same ASE `row.id`).** Relaxation
preserves atoms, so the chemical formula MUST be identical — validated 8/8 across the
full id range; use formula equality as a guard and skip/flag any mismatch.

## Per-row layout: everything is in `row.data` (`key_value_pairs` is EMPTY)

`row.toatoms()` gives the structure. No ASE `.energy` attribute. Fields under `row.data`:

### Energies & forces — `row.data['properties']['orb_properties']`
- `orb_energy_per_atom: float` (eV/atom)
- `orb_max_force: float`
- In `relaxed.db` this is the **relaxed** energy/force → our regression target
  (`energy_per_atom`).
- In `samples.db` this is the **unrelaxed** Orb energy → Orb-dependent, so a reference
  only (NOT a surrogate input, to keep the surrogate Orb-free).

### Relaxation quality (relaxed.db, top-level `row.data`)
- `geo_converged: bool` (~83% True among finals)
- `max_force: float`, `is_final: str` ('1' for all rows — not a useful filter)

### Geometry descriptors — `row.data['properties']['pyzeo_geometric_properties']`
All values are **strings** or the string `'None'`; cast to float, `'None'`/None → NaN.
Keys: `lcd, pld, dif, number_of_channels, number_of_pockets, av_volume_fraction,
av_cm3_per_g, nav_volume_fraction, nav_cm3_per_g, channel_volume_fraction,
pocket_volume_fraction, asa_m2_per_cm3, asa_m2_per_g, nasa_m2_per_cm3, nasa_m2_per_g,
channel_surface_area_fraction, pocket_surface_area_fraction`.
Use the **unrelaxed** (samples.db) geometry as features (the surrogate sees unrelaxed input).

### Validity flags — `row.data['metrics']` (present in samples.db)
- `smact_valid: bool`
- `no_atom_too_close: bool`
- `reconstruction_failed: bool`
- `mofchecker: dict` → aggregate `mofchecker['mofchecker_valid']: bool` (+ ~17 sub-flags)

### Topology — `row.data['topology']` (string)
**~78% are `ERROR` / `UNKNOWN` / `NA`.** Only ~22% have real RCSR codes (`sql, rna, hcb,
pcu, fcu, kgd, fes, ...`). The topology-holdout split must group only the valid-topology
subset across train/val/test; invalid-topology rows go to train (or are excluded from the
generalization test).

## Resulting decisions for the pipeline
- **Pair by row.id**, guard with formula equality.
- **Regression target** `energy_per_atom` = relaxed `orb_properties.orb_energy_per_atom`.
- **Stability label** (binary) = `mofchecker_valid AND smact_valid AND no_atom_too_close
  AND NOT reconstruction_failed AND geo_converged`. (Flags from samples.db; `geo_converged`
  from relaxed.db.) Sensitivity = which flags are included.
- **Features** = parsed `pyzeo_geometric_properties` (unrelaxed) + composition features.
  Exclude `orb_*` energies from features (Orb-free surrogate).
- **Topology split** on valid-topology subset only.
