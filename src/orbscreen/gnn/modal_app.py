"""Modal GPU app for training the OrbScreen crystal GNN.

Two entrypoints:
- `smoke`: cheap synthetic training on GPU (no data download) to verify the image,
  CUDA, torch/PyG, the W&B secret, and Volume writes work end-to-end.
- `train_model`: the real single-seed training on MofasaDB (downloads/builds on a Volume).

Run:
    uv run modal run src/orbscreen/gnn/modal_app.py --mode smoke
    uv run modal run src/orbscreen/gnn/modal_app.py --mode train --seed 0
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "torch_geometric>=2.6",
        "ase>=3.23",
        "pymatgen>=2024.5.1",
        "pandas>=2.2",
        "pyarrow>=16.0",
        "scikit-learn>=1.5",
        "scipy>=1.13",
        "wandb>=0.28.0",
        "huggingface_hub>=0.24",
    )
    .add_local_python_source("orbscreen")
)

app = modal.App("orbscreen-gnn", image=image)
vol = modal.Volume.from_name("orbscreen-data", create_if_missing=True)
WANDB = modal.Secret.from_name("wandb")


def _flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{prefix}{k}."))
        else:
            flat[prefix + k] = v
    return flat


@app.function(gpu="A10G", volumes={"/data": vol}, timeout=900, secrets=[WANDB])
def smoke() -> str:
    """Tiny synthetic GPU training run — verifies infra without any data download."""
    import os

    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from orbscreen.gnn.model import CrystalGNN
    from orbscreen.gnn.train import train_one_epoch

    assert torch.cuda.is_available(), "CUDA not available on this Modal GPU"
    device = "cuda"

    def g(z, e, s):
        n = len(z)
        ei = torch.tensor([[i for i in range(n)], [(i + 1) % n for i in range(n)]])
        d = Data(z=torch.tensor(z), edge_index=ei, edge_dist=torch.ones(ei.shape[1]), num_nodes=n)
        d.y_energy = torch.tensor([e], dtype=torch.float)
        d.y_stab = torch.tensor([s], dtype=torch.long)
        return d

    data = [g([1, 8], -6.0, 1), g([6, 1, 1], -7.0, 0)] * 16
    loader = DataLoader(data, batch_size=8)
    model = CrystalGNN(hidden=32, n_layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = [train_one_epoch(model, loader, opt, device) for _ in range(3)]

    import wandb

    wandb.init(project="orbscreen", name="smoke")
    wandb.log({"smoke_final_loss": losses[-1]})
    wandb.finish()

    os.makedirs("/data", exist_ok=True)
    torch.save({"ok": True, "losses": losses}, "/data/smoke.pt")
    vol.commit()
    return f"OK gpu={torch.cuda.get_device_name(0)} losses={[round(x, 4) for x in losses]}"


@app.function(gpu="A10G", volumes={"/data": vol}, timeout=4 * 60 * 60, secrets=[WANDB])
def train_model(seed: int = 0, config: dict | None = None, data_limit: int | None = None) -> str:
    """Real single-seed training on MofasaDB; writes a checkpoint to the Volume.

    data_limit caps the number of structures (for a cheap end-to-end check); a limited
    run uses its own parquet + graph cache so it never collides with the full dataset.
    """
    from pathlib import Path

    import torch
    from torch_geometric.loader import DataLoader

    from orbscreen.data import download
    from orbscreen.data.build import build_dataset
    from orbscreen.gnn.dataset import MofaGraphDataset
    from orbscreen.gnn.model import CrystalGNN
    from orbscreen.gnn.train import TrainConfig, evaluate, train_one_epoch

    cfg = TrainConfig(**(config or {}))
    torch.manual_seed(seed)

    paths = download.ensure_files(["samples.db", "relaxed.db"])
    suffix = f"_limit{data_limit}" if data_limit else ""
    parquet = f"/data/dataset{suffix}.parquet"
    cache = f"/data/cache{suffix}"
    if not Path(parquet).exists():
        build_dataset(paths["samples.db"], paths["relaxed.db"], parquet, limit=data_limit)
        vol.commit()

    samples = str(paths["samples.db"])
    train_ds = MofaGraphDataset(samples, parquet, "split_random", "train", cache_dir=cache)
    test_ds = MofaGraphDataset(samples, parquet, "split_random", "test", cache_dir=cache)
    vol.commit()

    device = "cuda"
    model = CrystalGNN(cfg.hidden, cfg.n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    import wandb

    wandb.init(project="orbscreen", name=f"gnn-seed{seed}", config=cfg.__dict__)
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, train_loader, opt, device)
        wandb.log({"epoch": epoch, "train_loss": loss})
    metrics = evaluate(model, test_loader, device)
    wandb.log({"test/" + k: v for k, v in _flatten(metrics).items()})
    wandb.finish()

    out = f"/data/ckpt_seed{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__, "metrics": metrics}, out)
    vol.commit()
    return f"saved {out} metrics={metrics}"


@app.local_entrypoint()
def main(mode: str = "smoke", seed: int = 0, data_limit: int = 0, epochs: int = 0):
    if mode == "smoke":
        print(smoke.remote())
    elif mode == "train":
        cfg = {"epochs": epochs} if epochs else None
        print(train_model.remote(seed=seed, config=cfg, data_limit=data_limit or None))
    else:
        raise SystemExit(f"unknown mode: {mode!r} (use 'smoke' or 'train')")
