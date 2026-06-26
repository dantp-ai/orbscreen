"""Modal GPU app for training the OrbScreen crystal GNN.

Two entrypoints:
- `smoke`: cheap synthetic training on GPU (no data download) to verify the image,
  CUDA, torch/PyG, the W&B secret, and Volume writes work end-to-end.
- `train_model`: the real single-seed training on MofasaDB (downloads/builds on a Volume).

Run:
    uv run modal run src/orbscreen/gnn/modal_app.py --mode smoke
    uv run modal run src/orbscreen/gnn/modal_app.py --mode train --seed 0
"""

import os
from pathlib import Path

import modal


def _local_hf_token() -> str:
    """Read the local HF token (env var or cached login) to authenticate downloads."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    cached = Path.home() / ".cache" / "huggingface" / "token"
    return cached.read_text().strip() if cached.exists() else ""


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
# Pass the local HF token (if any) so MofasaDB downloads are authenticated; empty -> anon.
_HF_TOKEN = _local_hf_token()
HF = modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN} if _HF_TOKEN else {})


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


@app.function(gpu="A10G", volumes={"/data": vol}, timeout=4 * 60 * 60, secrets=[WANDB, HF])
def train_model(seed: int = 0, config: dict | None = None, data_limit: int | None = None) -> str:
    """Real single-seed training on MofasaDB; writes a checkpoint to the Volume.

    data_limit caps the number of structures (for a cheap end-to-end check); a limited
    run uses its own parquet + graph cache so it never collides with the full dataset.
    """
    import torch
    from torch_geometric.loader import DataLoader

    from orbscreen.data import download
    from orbscreen.data.build import build_dataset
    from orbscreen.gnn.dataset import MofaGraphDataset
    from orbscreen.gnn.model import CrystalGNN
    from orbscreen.gnn.train import TrainConfig, evaluate, train_with_early_stopping

    cfg = TrainConfig(**(config or {}))
    torch.manual_seed(seed)

    suffix = f"_limit{data_limit}" if data_limit else ""
    parquet = f"/data/dataset{suffix}.parquet"
    cache = f"/data/cache{suffix}"
    # Only download the 4.5 GB source DBs if a graph cache is missing; ensemble runs reuse
    # the committed caches, so they skip the download (and the HF rate-limit warning) entirely.
    splits = ("train", "val", "test")
    caches_present = all(
        Path(f"{cache}/processed/graphs_split_random_{s}.pt").exists() for s in splits
    )
    samples = ""
    if not caches_present:
        paths = download.ensure_files(["samples.db", "relaxed.db"])
        samples = str(paths["samples.db"])
        if not Path(parquet).exists():
            build_dataset(paths["samples.db"], paths["relaxed.db"], parquet, limit=data_limit)
            vol.commit()

    train_ds = MofaGraphDataset(samples, parquet, "split_random", "train", cache_dir=cache)
    val_ds = MofaGraphDataset(samples, parquet, "split_random", "val", cache_dir=cache)
    test_ds = MofaGraphDataset(samples, parquet, "split_random", "test", cache_dir=cache)
    vol.commit()

    device = "cuda"
    model = CrystalGNN(cfg.hidden, cfg.n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    import wandb

    wandb.init(project="orbscreen", name=f"gnn-seed{seed}", config=cfg.__dict__)
    es = train_with_early_stopping(
        model, train_loader, val_loader, opt, device,
        max_epochs=cfg.epochs, patience=cfg.patience, w_stab=cfg.w_stab,
        log_fn=wandb.log,
    )
    metrics = evaluate(model, test_loader, device)
    wandb.log({"test/" + k: v for k, v in _flatten(metrics).items()})
    wandb.log({"best_epoch": es["best_epoch"], "epochs_run": es["epochs_run"]})
    wandb.finish()

    out = f"/data/ckpt_seed{seed}.pt"
    es_summary = {k: es[k] for k in ("best_epoch", "best_val_loss", "epochs_run", "stopped_early")}
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg.__dict__,
         "metrics": metrics, "early_stop": es_summary},
        out,
    )
    vol.commit()
    return f"saved {out} best_epoch={es['best_epoch']} epochs_run={es['epochs_run']} metrics={metrics}"


@app.function(gpu="A10G", memory=32768, volumes={"/data": vol}, timeout=4 * 60 * 60, secrets=[WANDB, HF])
def train_topology(seed: int = 0, config: dict | None = None) -> str:
    """Train on the topology-holdout split, reusing the random-split graph caches
    (re-sliced by topology in memory — no graph rebuild). Saves ckpt_topo_seed{seed}.pt.
    """
    import torch
    from torch_geometric.loader import DataLoader

    from orbscreen.gnn.dataset import reslice_split
    from orbscreen.gnn.model import CrystalGNN
    from orbscreen.gnn.train import TrainConfig, train_with_early_stopping

    cfg = TrainConfig(**(config or {}))
    torch.manual_seed(seed)
    parquet = "/data/dataset.parquet"
    cache = "/data/cache"
    train_graphs = reslice_split("", parquet, cache, "split_topology", "train")
    val_graphs = reslice_split("", parquet, cache, "split_topology", "val")

    device = "cuda"
    model = CrystalGNN(cfg.hidden, cfg.n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    import wandb

    wandb.init(project="orbscreen", name=f"gnn-topo-seed{seed}", config=cfg.__dict__)
    es = train_with_early_stopping(
        model,
        DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True),
        DataLoader(val_graphs, batch_size=cfg.batch_size),
        opt, device, cfg.epochs, cfg.patience, cfg.w_stab, log_fn=wandb.log,
    )
    wandb.finish()
    out = f"/data/ckpt_topo_seed{seed}.pt"
    es_summary = {k: es[k] for k in ("best_epoch", "best_val_loss", "epochs_run", "stopped_early")}
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__, "early_stop": es_summary}, out)
    vol.commit()
    return f"saved {out} best_epoch={es['best_epoch']} n_train={len(train_graphs)} n_val={len(val_graphs)}"


@app.function(gpu="A10G", memory=32768, volumes={"/data": vol}, timeout=60 * 60, secrets=[WANDB, HF])
def evaluate_models(split: str = "split_random") -> dict:
    """Evaluate the deep ensemble on a split's test set; write results JSON to the Volume.

    The ensemble is the full-data early-stopped checkpoints (ckpt_seed1..N); the limited
    5k ckpt_seed0 is excluded. For a split whose test-graph cache is absent, the source
    DBs are downloaded once to build it.
    """
    import glob
    import json

    from orbscreen.data import download
    from orbscreen.gnn.evaluate import evaluate_ensemble

    if split == "split_topology":
        ckpts = sorted(glob.glob("/data/ckpt_topo_seed*.pt"))
    else:
        ckpts = sorted(p for p in glob.glob("/data/ckpt_seed*.pt") if "ckpt_seed0.pt" not in p)
    if not ckpts:
        raise SystemExit(f"no ensemble checkpoints found on volume for {split}")

    cache = "/data/cache"
    parquet = "/data/dataset.parquet"
    samples = ""
    if not Path(f"{cache}/processed/graphs_{split}_test.pt").exists():
        paths = download.ensure_files(["samples.db", "relaxed.db"])
        samples = str(paths["samples.db"])

    res = evaluate_ensemble(ckpts, samples, parquet, split, cache_dir=cache)
    res["checkpoints"] = [p.rsplit("/", 1)[-1] for p in ckpts]
    Path(f"/data/results_gnn_{split}.json").write_text(json.dumps(res, indent=2))
    vol.commit()
    return res


@app.local_entrypoint()
def main(
    mode: str = "smoke", seed: int = 0, data_limit: int = 0,
    epochs: int = 0, patience: int = 0, split: str = "split_random",
):
    import json

    if mode == "smoke":
        print(smoke.remote())
    elif mode == "train":
        cfg = {}
        if epochs:
            cfg["epochs"] = epochs
        if patience:
            cfg["patience"] = patience
        print(train_model.remote(seed=seed, config=cfg or None, data_limit=data_limit or None))
    elif mode == "train_topology":
        cfg = {}
        if epochs:
            cfg["epochs"] = epochs
        if patience:
            cfg["patience"] = patience
        print(train_topology.remote(seed=seed, config=cfg or None))
    elif mode == "eval":
        print(json.dumps(evaluate_models.remote(split=split), indent=2))
    else:
        raise SystemExit(f"unknown mode: {mode!r} (use 'smoke', 'train', 'train_topology', or 'eval')")
