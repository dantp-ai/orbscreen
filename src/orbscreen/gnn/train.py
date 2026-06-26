from dataclasses import dataclass

import torch
import torch.nn.functional as F

from orbscreen.eval.metrics import (
    classification_metrics,
    enrichment_factor,
    precision_at_k,
    regression_metrics,
)


@dataclass
class TrainConfig:
    hidden: int = 128
    n_layers: int = 3
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 64
    w_stab: float = 1.0
    seed: int = 0


def multitask_loss(out, batch, w_stab: float = 1.0):
    e = F.mse_loss(out["energy"], batch.y_energy.view(-1))
    s = F.binary_cross_entropy_with_logits(out["stab_logit"], batch.y_stab.view(-1).float())
    return e + w_stab * s


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = multitask_loss(model(batch), batch)
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    import numpy as np

    model.eval()
    e_pred, e_true, p_stab, y_stab = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        e_pred += out["energy"].cpu().tolist()
        e_true += batch.y_energy.view(-1).cpu().tolist()
        p_stab += torch.sigmoid(out["stab_logit"]).cpu().tolist()
        y_stab += batch.y_stab.view(-1).cpu().tolist()
    y = np.array(y_stab)
    p = np.array(p_stab)
    k = max(1, int(0.1 * len(y)))
    return {
        "regression": regression_metrics(e_true, e_pred),
        "classification": classification_metrics(y, p),
        "ranking": {
            "base_rate": float(y.mean()),
            "precision_at_10pct": precision_at_k(y, p, k),
            "enrichment_at_10pct": enrichment_factor(y, p, k),
        },
    }
