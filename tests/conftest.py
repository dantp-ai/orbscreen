import pytest
from ase import Atoms
from ase.db import connect


def _atoms(n_extra: int = 0, displace: float = 0.0) -> Atoms:
    """Build a structure with 3 + n_extra atoms; identical n_extra => identical formula."""
    symbols = "H2O" + "H" * n_extra
    positions = [[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]] + [[1.5 + 0.3 * i, 0, 0] for i in range(n_extra)]
    atoms = Atoms(symbols, positions=positions, cell=[12, 12, 12], pbc=True)
    if displace:
        atoms.positions = atoms.positions + displace
    return atoms


@pytest.fixture
def fake_dbs(tmp_path):
    """Two tiny ASE DBs mirroring MofasaDB's nested layout, paired positionally by row.id.

    6 structures (distinct formulas). By construction only i=0 is 'stable'.
    Topologies: pcu, sql, ERROR, UNKNOWN, pcu, sql.
    """
    samples = connect(str(tmp_path / "samples.db"))
    relaxed = connect(str(tmp_path / "relaxed.db"))
    topos = ["pcu", "sql", "ERROR", "UNKNOWN", "pcu", "sql"]
    for i in range(6):
        sdata = {
            "topology": topos[i],
            "properties": {
                "orb_properties": {"orb_energy_per_atom": -6.0 - 0.1 * i, "orb_max_force": 0.2},
                "pyzeo_geometric_properties": {
                    "lcd": str(5.0 + i),
                    "pld": str(3.0 + i),
                    "dif": str(4.0 + i),
                    "av_cm3_per_g": str(0.1 * i),
                    "asa_m2_per_g": "None" if i == 0 else str(1000.0 + 100 * i),
                    "asa_m2_per_cm3": 0.0,
                    "number_of_channels": float(i % 2),
                },
            },
            "metrics": {
                "smact_valid": (i % 2 == 0),
                "no_atom_too_close": True,
                "reconstruction_failed": (i == 5),
                "mofchecker": {"mofchecker_valid": (i < 3)},
            },
        }
        samples.write(_atoms(n_extra=i), data=sdata)
        rdata = {
            "geo_converged": (i != 2),
            "properties": {"orb_properties": {"orb_energy_per_atom": -6.5 - 0.1 * i, "orb_max_force": 0.05}},
        }
        relaxed.write(_atoms(n_extra=i, displace=0.05), data=rdata)
    return {"samples": str(tmp_path / "samples.db"), "relaxed": str(tmp_path / "relaxed.db")}
