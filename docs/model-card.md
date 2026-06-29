# Model Card — OrbScreen GNN surrogate

## Overview
A graph neural network that predicts the **Orb-v3 relaxed-state stability** of a metal-organic
framework directly from its **unrelaxed** structure — skipping both the geometry relaxation and
the interatomic-potential evaluation. It is meant as a fast, Orb-free pre-filter for
high-throughput MOF screening, feeding a cascade where Orb-v3 confirms the top/uncertain
candidates.

## Architecture
- **`CrystalGNN`** (multi-task): atom-number embedding → Gaussian radial-basis expansion of
  interatomic distances → 3× `CGConv` message-passing layers → global mean pool → two heads:
  an **energy** head (regress Orb-v3 relaxed eV/atom) and a **stability** head (binary logit).
- **Input graph:** PBC-aware radius graph (6 Å cutoff) built from the unrelaxed structure with
  ASE neighbor lists (nodes = atomic numbers, edges = interatomic distances). The model is
  **Orb-free** — it never sees any Orb energy as input.
- **Uncertainty:** deep ensemble (multiple seeds); predictive mean + std, with calibration (ECE).

## Training data
- **MofasaDB** (Orbital Materials, CC-BY-4.0): 201,926 generated MOFs. Unrelaxed structures
  (`samples.db`) paired positionally to their Orb-v3-relaxed counterparts (`relaxed.db`).
- **Targets:** regression = Orb-v3 relaxed energy/atom; stability = validity-flag composite
  (`mofchecker_valid ∧ smact_valid ∧ no_atom_too_close ∧ geo_converged ∧ ¬reconstruction_failed`).
- **Splits:** random (80/10/10) and **topology-holdout** (RCSR frameworks partitioned so test
  frameworks are unseen; the ~78% of rows with ERROR/UNKNOWN/NA topology are training-only).

## Training procedure
- PyTorch + PyTorch Geometric on a Modal **A10G** GPU; Adam; multi-task loss = MSE(energy) +
  BCE(stability); **val-based early stopping** (patience) with **best-checkpoint** restore.
- Ensembles: 5 seeds (random split), 3 seeds (topology split). Converges in ~3–8 epochs.
- Experiment tracking in Weights & Biases.

## Performance (test set; deep ensemble)
| Metric | Random split (n=20,194) | Topology-holdout (n=2,270) |
|---|---|---|
| Energy MAE (eV/atom) | 0.036 | 0.029 |
| Energy Spearman | 0.992 | 0.989 |
| Stability AUROC | 0.932 | 0.956 |
| Stability AUPRC (base rate ~0.05) | 0.319 | 0.420 |
| Enrichment @ top-10% | 6.07× | 7.13× |
| Calibration (ECE) | 0.0064 | 0.0098 |

Baseline (gradient-boosted descriptors) for reference: MAE 0.076 both splits; enrichment
5.24× / 5.37×; AUROC 0.881 / 0.924. The GNN ensemble beats it on every metric on both splits.

## Throughput & cost (Phase 3)

Measured on Modal A10G GPU (201,926-MOF screen) and a 100-structure Orb-v3 benchmark:

| | Surrogate (this work) | Orb-v3 relaxation |
|---|---|---|
| Throughput (structs/s/GPU) | 149.1 | 0.211 |
| Cost - inference only (USD/million) | 2.05 | 1,447 |
| Cost - end-to-end incl. graph build (USD/million) | 10.29 | 1,447 |

A cost-accuracy cascade routing 20% of candidates (uncertainty policy) to Orb-v3 recovers 98.1% of Orb-v3's top-10% stable MOFs at 5.0x lower cost than relaxing everything.
For energy ranking, routing the top-ranked 15% (confirm-top-ranked policy) recovers 99.8% at 6.6x lower cost.

## Intended use
- **Pre-screening / triage:** rank large MOF libraries by predicted P(stable) to prioritise
  expensive Orb-v3 (or DFT) confirmation — a cost-aware cascade.
- Research / demonstration; not a substitute for Orb-v3 or DFT on candidates that matter.

## Limitations
- **Target is Orb-v3's verdict**, not DFT/experiment directly — this approximates Orb-v3
  (a surrogate of a surrogate); it inherits Orb-v3's biases.
- **Stability is a validity-flag composite**, not thermodynamic energy-above-hull (ill-defined
  for MOFs here); the threshold/flag choice is a modelling decision (see `docs/data-schema.md`).
- **Topology-holdout test is the valid-topology subset** (n=2,270) — a smaller, cleaner
  population than the random-split test, so absolute cross-split numbers are not directly
  comparable; the GNN-vs-baseline comparison on each split is fair (same test set).
- Trained on **Mofasa-generated** MOFs; behaviour on very different MOF distributions is untested.
- Early-stopped at a few epochs; not exhaustively hyperparameter-tuned.

## Licenses & attribution
- MofasaDB — CC-BY-4.0 (Orbital Materials).
- Orb-v3 — Apache-2.0 (Orbital Materials); arXiv:2504.06231.
