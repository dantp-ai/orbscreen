import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data

CUTOFF = 6.0


def structure_to_graph(atoms: Atoms, cutoff: float = CUTOFF) -> Data:
    """Build a PBC-aware radius graph from an ASE structure."""
    i, j, dvec = neighbor_list("ijD", atoms, cutoff)
    dist = np.linalg.norm(dvec, axis=1)
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
    edge_index = torch.tensor(np.vstack([i, j]), dtype=torch.long)
    edge_dist = torch.tensor(dist, dtype=torch.float)
    return Data(z=z, pos=pos, edge_index=edge_index, edge_dist=edge_dist, num_nodes=len(atoms))
