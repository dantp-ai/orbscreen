from orbscreen.gnn.dataset import MofaGraphDataset, reslice_split


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


def test_load_all_graphs_and_reslice_by_topology(fake_dbs_large, tmp_path):
    from orbscreen.data.build import build_dataset

    parquet = str(tmp_path / "ds.parquet")
    build_dataset(fake_dbs_large["samples"], fake_dbs_large["relaxed"], parquet)
    cache = str(tmp_path / "c")
    # build the three random-split caches (together they cover all structures)
    total = sum(
        len(MofaGraphDataset(fake_dbs_large["samples"], parquet, "split_random", v, cache_dir=cache))
        for v in ("train", "val", "test")
    )
    # re-slicing by topology partitions the same graphs (no rebuild): parts sum to the whole
    n = sum(len(reslice_split(fake_dbs_large["samples"], parquet, cache, "split_topology", v))
            for v in ("train", "val", "test"))
    assert total >= 1 and n == total
