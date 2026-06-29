import numpy as np
import torch

from orbscreen.gnn.model import CrystalGNN


def _load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = CrystalGNN(cfg["hidden"], cfg["n_layers"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def ensemble_predict(checkpoints, loader, device) -> dict:
    """Average predictions across an ensemble of checkpoints.

    Returns per-test-item mean/std for energy and P(stable), plus the targets.
    """
    models = [_load(c, device) for c in checkpoints]
    e_all, s_all, y_e, y_s, gids = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        e_k = np.stack([m(batch)["energy"].cpu().numpy() for m in models])
        s_k = np.stack([torch.sigmoid(m(batch)["stab_logit"]).cpu().numpy() for m in models])
        e_all.append(e_k)
        s_all.append(s_k)
        y_e += batch.y_energy.view(-1).cpu().tolist()
        y_s += batch.y_stab.view(-1).cpu().tolist()
        if hasattr(batch, "gid"):
            gids += batch.gid.view(-1).cpu().tolist()
    e = np.concatenate(e_all, axis=1)
    s = np.concatenate(s_all, axis=1)
    return {
        "energy_mean": e.mean(0), "energy_std": e.std(0),
        "stab_mean": s.mean(0), "stab_std": s.std(0),
        "y_energy": np.array(y_e), "y_stab": np.array(y_s),
        "gid": np.array(gids, dtype=int),
    }
