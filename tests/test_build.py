import pandas as pd

from orbscreen.data.build import build_dataset, build_records


def test_build_records_positional_pairing(fake_dbs):
    records, stats = build_records(fake_dbs["samples"], fake_dbs["relaxed"])
    assert stats["paired"] == 6 and stats["formula_mismatch"] == 0
    by_id = {r["id"]: r for r in records}
    # row id=1 (i=0): relaxed energy -6.5, geometry from samples, only-stable row
    assert by_id[1]["energy_per_atom"] == -6.5
    assert by_id[1]["geom_lcd"] == 5.0
    assert sum(r["stability"] for r in records) == 1
    assert by_id[1]["stability"] == 1


def test_build_dataset_writes_parquet(fake_dbs, tmp_path):
    out = tmp_path / "dataset.parquet"
    df, stats = build_dataset(fake_dbs["samples"], fake_dbs["relaxed"], out)
    assert out.exists()
    loaded = pd.read_parquet(out)
    for col in ["id", "energy_per_atom", "stability", "topology", "geom_pld",
                "split_random", "split_topology"]:
        assert col in loaded.columns
    assert len(loaded) == 6
