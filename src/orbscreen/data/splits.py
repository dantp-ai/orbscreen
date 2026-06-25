"""Train/val/test splits.

- random_split: split rows independently.
- topology_split: leakage-free generalisation test. Only ~22% of MofasaDB rows have a
  real RCSR topology; the rest (ERROR/UNKNOWN/NA) cannot test framework generalisation,
  so they are always assigned to 'train' and the valid-topology groups are partitioned
  across train/val/test (no topology appears in more than one split).
"""

import numpy as np
import pandas as pd

from orbscreen import config


def _assign(n_items: int, fracs, rng) -> np.ndarray:
    idx = rng.permutation(n_items)
    n_train = int(fracs[0] * n_items)
    n_val = int(fracs[1] * n_items)
    labels = np.empty(n_items, dtype=object)
    labels[idx[:n_train]] = "train"
    labels[idx[n_train : n_train + n_val]] = "val"
    labels[idx[n_train + n_val :]] = "test"
    return labels


def random_split(df: pd.DataFrame, seed: int = 0, fracs=(0.8, 0.1, 0.1)) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(_assign(len(df), fracs, rng), index=df.index)


def topology_split(df: pd.DataFrame, seed: int = 0, fracs=(0.8, 0.1, 0.1)) -> pd.Series:
    rng = np.random.default_rng(seed)
    topo = df["topology"]
    valid_mask = ~topo.isin(config.INVALID_TOPOLOGIES) & topo.notna()
    labels = np.full(len(df), "train", dtype=object)
    valid_topos = sorted(topo[valid_mask].unique())
    if valid_topos:
        group_label = dict(zip(valid_topos, _assign(len(valid_topos), fracs, rng)))
        pos = {idx: i for i, idx in enumerate(df.index)}
        for idx in df.index[valid_mask]:
            labels[pos[idx]] = group_label[topo.loc[idx]]
    return pd.Series(labels, index=df.index)
