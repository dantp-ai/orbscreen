# src/orbscreen/app/modal_app.py
"""Modal app serving the OrbScreen Gradio demo on CPU, scale-to-zero.

Deploy:  modal deploy src/orbscreen/app/modal_app.py
The ensemble checkpoints (ckpt_seed1..5) and screen_predictions.parquet are read from the
Volume; the demo loads them once per container, then serves requests. CPU only - single-
structure ensemble inference is milliseconds, so no GPU is needed.
"""

import os
from pathlib import Path

import modal

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "torch_geometric>=2.6",
        "ase>=3.23",
        "numpy>=1.26",
        "pandas>=2.2",
        "pyarrow>=16.0",
        "gradio~=5.7",
        "fastapi[standard]>=0.115",
    )
    .add_local_python_source("orbscreen")
)

app = modal.App("orbscreen-demo", image=web_image)
vol = modal.Volume.from_name("orbscreen-data", create_if_missing=False)


def _demo_creds() -> dict:
    """Read the demo login (DEMO_USER/DEMO_PASSWORD) from the local env or repo-root .env at
    DEPLOY time and ship them as an ephemeral Modal secret - mirrors the HF-token pattern, and
    keeps credentials out of git (.env is gitignored). Returns empty in the container (no .env
    there); the deploy-time values are what Modal injects at runtime."""
    user = os.environ.get("DEMO_USER", "")
    pw = os.environ.get("DEMO_PASSWORD", "")
    if not (user and pw):
        env_file = Path(__file__).resolve().parents[3] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                key, _, val = line.partition("=")
                if key.strip() == "DEMO_USER" and not user:
                    user = val.strip()
                elif key.strip() == "DEMO_PASSWORD" and not pw:
                    pw = val.strip()
    return {"DEMO_USER": user, "DEMO_PASSWORD": pw}


DEMO_AUTH = modal.Secret.from_dict(_demo_creds())


@app.function(volumes={"/data": vol}, secrets=[DEMO_AUTH], max_containers=1, scaledown_window=300)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """Serve the Gradio Blocks via FastAPI behind a username/password login.
    max_containers=1 for Gradio sticky sessions; min_containers defaults to 0 (scale-to-zero)."""
    import glob

    import pandas as pd
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    from orbscreen.app.demo import build_demo

    user, pw = os.environ.get("DEMO_USER", ""), os.environ.get("DEMO_PASSWORD", "")
    if not (user and pw):
        raise RuntimeError("DEMO_USER/DEMO_PASSWORD not set - add them to .env before deploying")
    ckpts = sorted(p for p in glob.glob("/data/ckpt_seed*.pt") if "ckpt_seed0.pt" not in p)
    if not ckpts:
        raise RuntimeError("no ensemble checkpoints (ckpt_seed*.pt) on the Volume")
    preds = pd.read_parquet("/data/screen_predictions.parquet")
    demo = build_demo(preds, ckpts)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/", auth=(user, pw))
