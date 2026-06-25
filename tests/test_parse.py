import math

from ase import Atoms
from ase.db import connect

from orbscreen.data.parse import nested, relaxed_fields, sample_fields, to_float


def test_to_float_handles_strings_lists_and_none():
    assert to_float("2.61") == 2.61
    assert to_float([-6.54]) == -6.54
    assert math.isnan(to_float("None"))
    assert math.isnan(to_float(None))
    assert math.isnan(to_float("not_a_number"))


def test_nested_lookup():
    d = {"a": {"b": {"c": 5}}}
    assert nested(d, "a", "b", "c") == 5
    assert nested(d, "a", "x", default=-1) == -1


def test_sample_fields(fake_dbs):
    row = next(iter(connect(fake_dbs["samples"]).select("id=1")))
    f = sample_fields(row)
    assert f["topology"] == "pcu"
    assert f["geometry"]["lcd"] == 5.0
    assert math.isnan(f["geometry"]["asa_m2_per_g"])  # 'None' sentinel -> NaN
    assert f["smact_valid"] is True and f["mofchecker_valid"] is True


def test_relaxed_fields(fake_dbs):
    row = next(iter(connect(fake_dbs["relaxed"]).select("id=1")))
    f = relaxed_fields(row)
    assert f["energy_per_atom"] == -6.5  # relaxed orb energy/atom for i=0
    assert f["geo_converged"] is True


def test_sample_fields_handles_string_subrecords(tmp_path):
    # MofasaDB stores failed pyzeo/mofchecker as the string 'None'
    db = connect(str(tmp_path / "s.db"))
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]], cell=[10, 10, 10], pbc=True)
    db.write(
        atoms,
        data={
            "topology": "ERROR",
            "properties": {"orb_properties": {"orb_energy_per_atom": -6.0},
                           "pyzeo_geometric_properties": "None"},
            "metrics": {"smact_valid": False, "mofchecker": "None"},
        },
    )
    f = sample_fields(next(iter(db.select("id=1"))))
    assert all(math.isnan(v) for v in f["geometry"].values())
    assert f["mofchecker_valid"] is False
