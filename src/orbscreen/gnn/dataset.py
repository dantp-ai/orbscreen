from pathlib import Path

import pandas as pd
import torch
from ase.db import connect
from torch.utils.data import ConcatDataset
from torch_geometric.data import InMemoryDataset

from orbscreen.gnn.graph import CUTOFF, structure_to_graph


class MofaGraphDataset(InMemoryDataset):
    def __init__(self, samples_db, parquet, split, split_value, cutoff=CUTOFF, cache_dir=None):
        self._samples_db = samples_db
        self._parquet = parquet
        self._split = split
        self._split_value = split_value
        self._cutoff = cutoff
        self._cache = Path(cache_dir or ".graph_cache")
        super().__init__(root=str(self._cache))
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"graphs_{self._split}_{self._split_value}.pt"]

    def process(self):
        df = pd.read_parquet(self._parquet)
        sub = df[df[self._split] == self._split_value]
        targets = {int(r.id): (float(r.energy_per_atom), int(r.stability), str(r.formula))
                   for r in sub.itertuples(index=False)}
        db = connect(self._samples_db)
        data_list = []
        for row in db.select():
            t = targets.get(int(row.id))
            if t is None:
                continue
            y_energy, y_stab, formula = t
            if row.formula != formula:
                continue
            g = structure_to_graph(row.toatoms(), self._cutoff)
            g.y_energy = torch.tensor([y_energy], dtype=torch.float)
            g.y_stab = torch.tensor([y_stab], dtype=torch.long)
            g.gid = torch.tensor([int(row.id)], dtype=torch.long)
            data_list.append(g)
        self.save(data_list, self.processed_paths[0])


def _graph_ids(ds: MofaGraphDataset) -> list[int]:
    """All gids of a cached dataset, read from the collated tensor (no per-graph materialize)."""
    return [int(x) for x in ds._data.gid.view(-1).tolist()]


def reslice_split(samples_db, parquet, cache_dir, target_split, value, source_split="split_random"):
    """Build a (collated) dataset for `target_split == value` by reusing the already-built
    `source_split` graph caches — re-slicing by gid instead of rebuilding ~200k graphs.

    The source split's train/val/test caches together cover every structure, so any other
    split can be assembled from them. Subsets stay collated (memory-efficient) via PyG
    index-selection; a ConcatDataset stitches the parts back together.
    """
    df = pd.read_parquet(parquet, columns=["id", target_split])
    keep = set(df.loc[df[target_split] == value, "id"].astype(int))
    parts = []
    for sv in ("train", "val", "test"):
        ds = MofaGraphDataset(samples_db, parquet, source_split, sv, cache_dir=cache_dir)
        idx = [i for i, gid in enumerate(_graph_ids(ds)) if gid in keep]
        if idx:
            parts.append(ds[idx])
    return ConcatDataset(parts) if parts else []
