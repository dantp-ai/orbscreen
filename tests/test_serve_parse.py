import pytest
from ase import Atoms
from ase.io import write

from orbscreen.serve.parse import read_structure


def test_read_structure_roundtrips_a_cif(tmp_path):
    a = Atoms("H2O", positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]],
              cell=[10, 10, 10], pbc=True)
    p = tmp_path / "s.cif"
    write(str(p), a)
    got = read_structure(str(p))
    assert isinstance(got, Atoms)
    assert len(got) == 3


def test_read_structure_rejects_garbage(tmp_path):
    p = tmp_path / "bad.cif"
    p.write_text("this is not a structure file")
    with pytest.raises(ValueError):
        read_structure(str(p))
