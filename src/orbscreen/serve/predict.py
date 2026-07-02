"""Target-free single-structure inference for the demo's serving path.

Distinct from gnn.ensemble.ensemble_predict (which reads y_energy/y_stab targets for eval);
the serving path has no labels. Reuses the same graph builder, model, and checkpoint loader.
"""

import numpy as np
import torch
from torch_geometric.data import Batch

from orbscreen.gnn.ensemble import _load
from orbscreen.gnn.graph import structure_to_graph


@torch.no_grad()
def predict_structure(atoms, checkpoints, device: str = "cpu") -> dict:
    """Run the deep ensemble on one ASE structure; return mean/std for P(stable) and energy/atom."""
    graph = structure_to_graph(atoms)
    if graph.edge_index.shape[1] == 0:
        raise ValueError(
            "Structure has no atoms within the 6 A cutoff - check the unit cell and units."
        )
    batch = Batch.from_data_list([graph]).to(device)
    models = [_load(c, device) for c in checkpoints]
    energies, probs = [], []
    for m in models:
        out = m(batch)
        energies.append(float(out["energy"].cpu().item()))
        probs.append(float(torch.sigmoid(out["stab_logit"]).cpu().item()))
    e, p = np.array(energies), np.array(probs)
    return {
        "p_stable": float(p.mean()),
        "p_stable_std": float(p.std()),
        "energy_pred": float(e.mean()),
        "energy_std": float(e.std()),
    }
