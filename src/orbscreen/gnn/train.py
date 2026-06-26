import copy
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
    epochs: int = 30  # max epochs (early stopping may end sooner)
    batch_size: int = 64
    w_stab: float = 1.0
    seed: int = 0
    patience: int = 5  # early-stopping patience on validation loss


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
def eval_loss(model, loader, device, w_stab: float = 1.0) -> float:
    """Mean multi-task loss over a loader (used as the early-stopping signal)."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        total += multitask_loss(model(batch), batch, w_stab).item() * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


def train_with_early_stopping(
    model, train_loader, val_loader, optimizer, device,
    max_epochs: int, patience: int, w_stab: float = 1.0, log_fn=None,
) -> dict:
    """Train up to max_epochs, stopping when val loss hasn't improved for `patience`
    epochs; restore and return the best (lowest val-loss) weights.

    log_fn, if given, is called with a per-epoch dict (epoch, train_loss, val_loss).
    """
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    bad = 0
    history = []
    for epoch in range(max_epochs):
        tl = train_one_epoch(model, train_loader, optimizer, device)
        vl = eval_loss(model, val_loader, device, w_stab)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl})
        if log_fn is not None:
            log_fn(history[-1])
        if vl < best_val - 1e-4:
            best_val, best_state, best_epoch, bad = vl, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "stopped_early": len(history) < max_epochs,
        "history": history,
    }


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
