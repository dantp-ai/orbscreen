import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from orbscreen.gnn.model import CrystalGNN
from orbscreen.gnn.train import eval_loss, train_one_epoch, train_with_early_stopping


def _toy_graph(z, y_e, y_s):
    n = len(z)
    ei = torch.tensor([[i for i in range(n)], [(i + 1) % n for i in range(n)]])
    g = Data(z=torch.tensor(z), edge_index=ei,
             edge_dist=torch.ones(ei.shape[1]), num_nodes=n)
    g.y_energy = torch.tensor([y_e], dtype=torch.float)
    g.y_stab = torch.tensor([y_s], dtype=torch.long)
    return g


def test_training_reduces_loss():
    data = [_toy_graph([1, 8], -6.0, 1), _toy_graph([6, 1, 1], -7.0, 0)] * 8
    loader = DataLoader(data, batch_size=4)
    model = CrystalGNN(hidden=32, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = train_one_epoch(model, loader, opt, "cpu")
    for _ in range(20):
        last = train_one_epoch(model, loader, opt, "cpu")
    assert last < first


def test_eval_loss_returns_float():
    data = [_toy_graph([1, 8], -6.0, 1), _toy_graph([6, 1, 1], -7.0, 0)] * 4
    loader = DataLoader(data, batch_size=4)
    model = CrystalGNN(hidden=16, n_layers=2)
    assert eval_loss(model, loader, "cpu") >= 0.0


def test_early_stopping_stops_and_restores_best():
    data = [_toy_graph([1, 8], -6.0, 1), _toy_graph([6, 1, 1], -7.0, 0)] * 8
    train_loader = DataLoader(data, batch_size=4)
    val_loader = DataLoader(data, batch_size=4)
    model = CrystalGNN(hidden=16, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    res = train_with_early_stopping(
        model, train_loader, val_loader, opt, "cpu", max_epochs=50, patience=3
    )
    # overfittable tiny data: val loss plateaus, so it should stop before max_epochs
    assert res["stopped_early"] is True
    assert res["epochs_run"] < 50
    assert 0 <= res["best_epoch"] < res["epochs_run"]
    assert len(res["history"]) == res["epochs_run"]
