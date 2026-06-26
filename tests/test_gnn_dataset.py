from orbscreen.gnn.dataset import MofaGraphDataset


def test_dataset_yields_graphs_with_targets(fake_dataset, tmp_path):
    ds = MofaGraphDataset(
        fake_dataset["samples"], fake_dataset["parquet"],
        split="split_random", split_value="train", cache_dir=str(tmp_path / "c"),
    )
    assert len(ds) >= 1
    g = ds[0]
    assert g.z.ndim == 1
    assert hasattr(g, "y_energy") and hasattr(g, "y_stab")
    assert g.edge_index.shape[0] == 2


def test_dataset_caches(fake_dataset, tmp_path):
    cache = str(tmp_path / "c")
    a = MofaGraphDataset(fake_dataset["samples"], fake_dataset["parquet"],
                         split="split_random", split_value="train", cache_dir=cache)
    b = MofaGraphDataset(fake_dataset["samples"], fake_dataset["parquet"],
                         split="split_random", split_value="train", cache_dir=cache)
    assert len(a) == len(b)
