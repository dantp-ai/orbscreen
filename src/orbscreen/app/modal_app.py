# src/orbscreen/app/modal_app.py
"""Modal app serving the OrbScreen Gradio demo on CPU, scale-to-zero.

Deploy:  modal deploy src/orbscreen/app/modal_app.py
The ensemble checkpoints (ckpt_seed1..5) and screen_predictions.parquet are read from the
Volume; the demo loads them once per container, then serves requests. CPU only - single-
structure ensemble inference is milliseconds, so no GPU is needed.
"""

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


@app.function(volumes={"/data": vol}, max_containers=1, scaledown_window=300)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """Serve the Gradio Blocks via FastAPI. max_containers=1 for Gradio sticky sessions;
    min_containers defaults to 0 (scale-to-zero); scaledown_window keeps it warm 5 min."""
    import glob

    import pandas as pd
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    from orbscreen.app.demo import build_demo

    ckpts = sorted(p for p in glob.glob("/data/ckpt_seed*.pt") if "ckpt_seed0.pt" not in p)
    if not ckpts:
        raise RuntimeError("no ensemble checkpoints (ckpt_seed*.pt) on the Volume")
    preds = pd.read_parquet("/data/screen_predictions.parquet")
    demo = build_demo(preds, ckpts)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
