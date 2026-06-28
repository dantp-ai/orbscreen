import pandas as pd
import torch

from orbscreen.data.build import build_dataset
from orbscreen.gnn.dataset import MofaGraphDataset
from orbscreen.gnn.model import CrystalGNN
from orbscreen.screen.inference import screen_corpus


def _tiny_checkpoint(tmp_path):
    model = CrystalGNN(hidden=16, n_layers=2)
    path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": model.state_dict(), "config": {"hidden": 16, "n_layers": 2}}, path)
    return str(path)


def test_screen_corpus_covers_all_ids_and_carries_descriptors(fake_dbs_large, tmp_path):
    parquet = str(tmp_path / "ds.parquet")
    build_dataset(fake_dbs_large["samples"], fake_dbs_large["relaxed"], parquet)
    cache = str(tmp_path / "c")
    # build the three random-split caches (together they cover the whole corpus)
    for v in ("train", "val", "test"):
        MofaGraphDataset(fake_dbs_large["samples"], parquet, "split_random", v, cache_dir=cache)

    ckpt = _tiny_checkpoint(tmp_path)
    df, timing = screen_corpus([ckpt], parquet, cache,
                               samples_db=fake_dbs_large["samples"], device="cpu")

    meta = pd.read_parquet(parquet)
    assert len(df) == len(meta)
    assert set(df["gid"]) == set(meta["id"].astype(int))
    required = {"gid", "p_stable", "p_stable_std", "energy_pred", "energy_std",
               "y_stab", "y_energy", "geom_lcd"}
    assert required <= set(df.columns)
    assert timing["n"] == len(df)
    assert timing["throughput_per_sec"] >= 0.0
    # y_stab in the predictions matches the dataset label for the same id
    merged = df.merge(meta[["id", "stability"]], left_on="gid", right_on="id")
    assert (merged["y_stab"] == merged["stability"]).all()
