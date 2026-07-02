import torch
from ase import Atoms

from orbscreen.gnn.model import CrystalGNN
from orbscreen.serve.predict import predict_structure


def _tiny_ckpt(tmp_path, name):
    model = CrystalGNN(hidden=16, n_layers=2)
    path = tmp_path / name
    torch.save({"state_dict": model.state_dict(), "config": {"hidden": 16, "n_layers": 2}}, path)
    return str(path)


def test_predict_structure_returns_calibrated_keys(tmp_path):
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]],
                  cell=[10, 10, 10], pbc=True)
    ckpts = [_tiny_ckpt(tmp_path, "a.pt"), _tiny_ckpt(tmp_path, "b.pt")]
    out = predict_structure(atoms, ckpts, device="cpu")
    assert set(out) == {"p_stable", "p_stable_std", "energy_pred", "energy_std"}
    assert 0.0 <= out["p_stable"] <= 1.0
    assert out["p_stable_std"] >= 0.0 and out["energy_std"] >= 0.0
    assert all(isinstance(v, float) for v in out.values())


def test_predict_structure_rejects_isolated_atoms(tmp_path):
    # atoms far apart in a huge cell -> no neighbors within the 6 A cutoff
    atoms = Atoms("H2", positions=[[0, 0, 0], [50, 50, 50]], cell=[100, 100, 100], pbc=True)
    ckpts = [_tiny_ckpt(tmp_path, "a.pt")]
    import pytest
    with pytest.raises(ValueError):
        predict_structure(atoms, ckpts, device="cpu")
