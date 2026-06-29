"""Full-corpus batched ensemble inference (the Phase 3 "screen").

Reuses the already-built split_random graph caches (train+val+test together cover the whole
corpus) so nothing is rebuilt, and aligns every prediction to dataset.parquet by integer id.
"""

import time

import pandas as pd
import torch
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from orbscreen.gnn.dataset import MofaGraphDataset
from orbscreen.gnn.ensemble import ensemble_predict

# Geometry/composition descriptors carried through for the Phase 4 carbon-capture score.
CARRIED_DESCRIPTORS = [
    "topology", "n_atoms",
    "geom_lcd", "geom_pld", "geom_dif",
    "geom_av_cm3_per_g", "geom_asa_m2_per_g", "geom_asa_m2_per_cm3",
]


def load_corpus_graphs(
    parquet: str,
    cache_dir: str,
    samples_db: str = "",
    source_split: str = "split_random",
) -> ConcatDataset:
    """The whole corpus = the source split's train+val+test caches concatenated (no rebuild)."""
    parts = [MofaGraphDataset(samples_db, parquet, source_split, v, cache_dir=cache_dir)
             for v in ("train", "val", "test")]
    return ConcatDataset(parts)


def screen_corpus(
    checkpoints: list[str],
    parquet: str,
    cache_dir: str,
    *,
    samples_db: str = "",
    device: str | None = None,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, dict]:
    """Run the deep ensemble over the full corpus; return (DataFrame, timing).

    DataFrame is aligned to dataset.parquet by `gid` (== id) and carries CARRIED_DESCRIPTORS.
    timing = {"n", "seconds", "throughput_per_sec"} for the inference pass (GPU-only, cached graphs).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    graphs = load_corpus_graphs(parquet, cache_dir, samples_db=samples_db)
    loader = DataLoader(graphs, batch_size=batch_size)
    t0 = time.perf_counter()
    pred = ensemble_predict(checkpoints, loader, device)
    seconds = time.perf_counter() - t0
    df = pd.DataFrame({
        "gid": pred["gid"],
        "p_stable": pred["stab_mean"], "p_stable_std": pred["stab_std"],
        "energy_pred": pred["energy_mean"], "energy_std": pred["energy_std"],
        "y_stab": pred["y_stab"].astype(int), "y_energy": pred["y_energy"],
    })
    meta = pd.read_parquet(parquet, columns=["id"] + CARRIED_DESCRIPTORS)
    df = df.merge(meta, left_on="gid", right_on="id", how="left").drop(columns=["id"])
    n = len(df)
    timing = {"n": n, "seconds": seconds,
              "throughput_per_sec": (n / seconds) if seconds > 0 else 0.0}
    return df, timing
