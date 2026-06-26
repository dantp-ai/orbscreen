import torch
from torch_geometric.data import Batch, Data

from orbscreen.gnn.model import CrystalGNN


def _toy_batch():
    g1 = Data(z=torch.tensor([1, 8]), edge_index=torch.tensor([[0, 1], [1, 0]]),
              edge_dist=torch.tensor([1.0, 1.0]), num_nodes=2)
    g2 = Data(z=torch.tensor([6, 1, 1]), edge_index=torch.tensor([[0, 1, 2], [1, 0, 0]]),
              edge_dist=torch.tensor([1.1, 1.1, 1.2]), num_nodes=3)
    return Batch.from_data_list([g1, g2])


def test_forward_shapes():
    model = CrystalGNN(hidden=32, n_layers=2)
    out = model(_toy_batch())
    assert out["energy"].shape == (2,)
    assert out["stab_logit"].shape == (2,)


def test_gradients_flow():
    model = CrystalGNN(hidden=32, n_layers=2)
    out = model(_toy_batch())
    loss = out["energy"].sum() + out["stab_logit"].sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
