# OrbScreen

A fast surrogate that predicts whether a metal-organic framework (MOF) is stable, as judged by Orbital Materials' Orb-v3 relaxation, directly from the unrelaxed structure.
It turns an expensive simulation step into a millisecond inference, and falls back to real Orb-v3 only for the candidates that need it.

## Why

Generative models produce hundreds of thousands of candidate MOFs, and evaluating them is the bottleneck.
Orb-v3 geometry relaxation is accurate but costly per structure.
OrbScreen learns Orb-v3's verdict from [MofasaDB](https://huggingface.co/datasets/Orbital-Materials/MofasaDB) (201,926 MOFs) so the full corpus can be screened cheaply.

## Results

Deep-ensemble crystal GNN vs a descriptor baseline, on a random split and on a topology-holdout split (frameworks unseen in training):

| Test set | Energy MAE (eV/atom) | Stability AUROC | Enrichment @ top-10% | Calibration (ECE) |
|---|---|---|---|---|
| Random: baseline -> GNN | 0.076 -> **0.036** | 0.88 -> **0.93** | 5.2x -> **6.1x** | **0.006** |
| Topology holdout: baseline -> GNN | 0.076 -> **0.029** | 0.92 -> **0.96** | 5.4x -> **7.1x** | **0.010** |

At scale, measured on the same A10G GPU:

- The surrogate screens ~149 structures/s (**~2 USD per million**) vs ~0.2/s for Orb-v3 relaxation (**~1,450 USD per million**), roughly 700x cheaper.
- A cost-accuracy cascade sends only the most uncertain or top-ranked candidates to Orb-v3. Routing 15-20% of candidates recovers **98-99.8%** of Orb-v3's top-10% at **5-6.6x lower cost** than relaxing everything.

Details: [`docs/model-card.md`](docs/model-card.md), `results*.json`, `cascade.json`, and the `cascade_*.png` Pareto plots.

## Demo

A password-protected Gradio app on Modal (CPU, scale-to-zero).
Upload a CIF/POSCAR or pick a screened MOF to get P(stable) with uncertainty and predicted energy.
A leaderboard tab ranks screened MOFs by a carbon-capture proxy score.

![OrbScreen demo](figures/3.png)

## How it works

1. **Data**: pair MofasaDB's unrelaxed and Orb-v3-relaxed structures; targets are relaxed energy/atom and a validity-flag stability label.
2. **Model**: a multi-task `CGConv` crystal GNN on the unrelaxed periodic graph, trained as a 5-model deep ensemble for calibrated uncertainty.
3. **Screen**: batch inference over all 201,926 MOFs, plus the cost model and cascade analysis.
4. **Serve**: single-structure inference behind a Gradio UI on Modal.

## Quickstart

```bash
uv sync --extra dev --extra gnn
uv run pytest

# Build the dataset (downloads ~4.5 GB from MofasaDB on first run) and run the baseline
uv run orbscreen build --out data/dataset.parquet
uv run orbscreen baseline --data data/dataset.parquet --out results.json
```

GPU steps run on [Modal](https://modal.com):

```bash
modal run src/orbscreen/gnn/modal_app.py --mode train --seed 1              # train one ensemble member
modal run src/orbscreen/gnn/modal_app.py --mode eval --split split_random   # evaluate the ensemble
modal run src/orbscreen/gnn/modal_app.py --mode screen                       # screen the full corpus
modal run src/orbscreen/gnn/modal_app.py --mode benchmark_orb --sample 100   # time real Orb-v3
uv run orbscreen cascade --predictions screen_predictions.parquet \
  --screen-timing screen_timing.json --orb-benchmark benchmark_orb.json
```

Deploy the demo (set `DEMO_USER` and `DEMO_PASSWORD` in a local `.env` first):

```bash
uv sync --extra dev --extra gnn --extra app
modal deploy src/orbscreen/app/modal_app.py
```

## Layout

```
src/orbscreen/
  data/       MofasaDB download, parsing, targets, splits
  features/   descriptor features   (baseline)
  models/     gradient-boosted baseline
  gnn/        CrystalGNN, training, ensemble, Modal GPU app
  screen/     full-corpus screen, cost model, cascade, carbon score
  serve/      single-structure inference and file parsing
  app/        Gradio demo and Modal serving
  eval/       metrics
```

## Caveats

- Stability is a pragmatic validity-flag composite, not energy-above-hull, which is ill-defined for MOFs.
- The carbon-capture score is a geometric proxy (stability x CO2-accessible pores x surface area), not an adsorption simulation.
- The topology-holdout test covers only MOFs with a resolved topology (n=2,270), so compare models within a split, not across splits.

## License

Built on MofasaDB (CC-BY-4.0) and Orb-v3 (Apache-2.0). See [`ATTRIBUTION.md`](ATTRIBUTION.md).
