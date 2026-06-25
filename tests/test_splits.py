import pandas as pd

from orbscreen.data.splits import random_split, topology_split


def _df(n=120):
    topos = ["pcu", "dia", "sql", "bcu", "fcu", "kgd"]
    return pd.DataFrame(
        {"id": list(range(n)), "topology": [topos[i % len(topos)] for i in range(n)]}
    )


def test_random_split_labels_and_size():
    s = random_split(_df())
    assert set(s.unique()) <= {"train", "val", "test"}
    assert len(s) == 120


def test_topology_split_no_leakage():
    df = _df().assign(split=topology_split(_df()).values)
    by_topo = df.groupby("topology")["split"].nunique()
    assert (by_topo == 1).all()


def test_topology_split_invalid_go_to_train():
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "topology": ["pcu", "sql", "ERROR", "UNKNOWN", "NA", None]}
    )
    s = topology_split(df)
    # rows 3..6 have invalid topology -> always train
    assert list(s.iloc[2:]) == ["train", "train", "train", "train"]
