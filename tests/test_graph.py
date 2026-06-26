import numpy as np
from ase import Atoms

from orbscreen.gnn.graph import structure_to_graph


def _two_atoms_periodic():
    # two atoms 2 A apart in a 4 A cell -> PBC images create extra neighbors within 6 A
    return Atoms("H2", positions=[[0, 0, 0], [2, 0, 0]], cell=[4, 4, 4], pbc=True)


def test_graph_has_expected_tensors():
    g = structure_to_graph(_two_atoms_periodic(), cutoff=6.0)
    assert g.z.tolist() == [1, 1]
    assert g.pos.shape == (2, 3)
    assert g.edge_index.shape[0] == 2
    assert g.edge_index.shape[1] == g.edge_dist.shape[0]
    assert g.edge_index.shape[1] > 0  # PBC produces neighbors


def test_graph_distances_positive_and_within_cutoff():
    g = structure_to_graph(_two_atoms_periodic(), cutoff=6.0)
    d = g.edge_dist.numpy()
    assert np.all(d > 0) and np.all(d <= 6.0 + 1e-6)
