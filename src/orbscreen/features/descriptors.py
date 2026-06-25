"""Build the feature matrix used by descriptor-based models (Phase 1 baseline).

Features:
- All ``geom_*`` columns present in the DataFrame (real geometry descriptors from pyzeo).
- ``n_atoms`` (scalar structural property).
- Fraction-weighted elemental composition statistics from ``formula``.

Columns that are targets, Orb-dependent inputs, validity flags, or split labels are
intentionally excluded by only selecting the columns above.
"""

import pandas as pd

from orbscreen.features.composition import composition_features


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(features_df, feature_names)`` for all rows in *df*.

    - Geometry: every column whose name starts with ``"geom_"`` plus ``"n_atoms"``.
    - Composition: the output of :func:`composition_features` for each ``formula``.
    - Non-finite values are filled with the column median then 0.0.
    - ``orb_energy_unrelaxed``, ``energy_per_atom``, ``stability``, split columns, and
      boolean flag columns are never included.
    """
    geom_cols = [c for c in df.columns if c.startswith("geom_")]
    structural_cols = ["n_atoms"] + geom_cols

    geom = df[[c for c in structural_cols if c in df.columns]].reset_index(drop=True)
    comp = pd.DataFrame(
        [composition_features(f) for f in df["formula"]]
    ).reset_index(drop=True)

    X = pd.concat([geom, comp], axis=1).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X, list(X.columns)
