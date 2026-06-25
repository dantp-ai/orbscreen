# OrbScreen

A fast, Orb-free surrogate that predicts Orb-v3 relaxed-state MOF stability directly from unrelaxed structures, enabling high-throughput screening of metal-organic frameworks. Built on Orbital Materials' open Orb-v3 model and MofasaDB dataset.

## Quickstart

```bash
uv sync --extra dev
uv run pytest

# Build the dataset from MofasaDB (downloads ~4.5 GB of ASE DBs on first run)
uv run orbscreen build --out data/dataset.parquet
# Train + evaluate the descriptor baseline on both splits
uv run orbscreen baseline --data data/dataset.parquet
```

## Data

MofasaDB ships 201,926 generated MOFs as two ASE DBs (`samples.db` unrelaxed,
`relaxed.db` Orb-v3-relaxed). The pipeline pairs them **positionally by row id**
(neither `structure_id` nor `mofid` is a unique key — verified by formula match across
all 201,926 rows), extracts the relaxed Orb-v3 energy/atom as the regression target and a
validity-flag-based stability label, and writes a unified Parquet with random and
leakage-free topology-holdout splits. See `docs/data-schema.md`.

## Phase 1 baseline results

Descriptor baseline (gradient-boosted trees on pyzeo geometry + composition features),
evaluated on the held-out test split. Predicting the **relaxed** state from the
**unrelaxed** structure, with no Orb evaluation at inference:

| Metric | random split | topology holdout |
|---|---|---|
| Energy/atom MAE (eV) | 0.076 | 0.077 |
| Energy/atom Spearman | 0.94 | 0.92 |
| Stability AUROC | 0.88 | 0.93 |
| Stability AUPRC (base rate ~0.05) | 0.26 | 0.32 |
| Enrichment @ top-10% | **5.3×** | **6.3×** |

Screening the top 10% by predicted P(stable) recovers stable MOFs at ~5–6× the base
rate, and the topology-holdout (unseen frameworks) holds up — the descriptors capture
transferable stability signal. This is the floor the learned GNN surrogate must beat.

See `ATTRIBUTION.md` for data/model licenses.
