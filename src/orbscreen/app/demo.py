"""Gradio Blocks for the OrbScreen demo: predict a MOF (upload or pick) + carbon-capture leaderboard."""

import gradio as gr
import pandas as pd

from orbscreen.screen.carbon import rank_corpus, score_corpus
from orbscreen.serve.parse import read_structure
from orbscreen.serve.predict import predict_structure

_DESC = (
    "Predict Orb-v3 relaxed-state MOF stability from an *unrelaxed* structure - no Orb-v3 "
    "evaluation at inference. Upload a CIF/POSCAR, or pick a screened MOF by id. The "
    "carbon-capture score is a geometric proxy (P(stable) x PLD>=3.3A gate x surface area), "
    "shown for screened MOFs only."
)


def build_demo(predictions_df: pd.DataFrame, checkpoints: list) -> gr.Blocks:
    """Build the demo UI around a screened-predictions table and the ensemble checkpoints."""
    scored = score_corpus(predictions_df).set_index("gid")
    leaderboard = rank_corpus(predictions_df, top_n=50)
    ids = [int(g) for g in predictions_df["gid"].tolist()]

    def on_upload(file) -> str:
        if file is None:
            return "Upload a CIF or POSCAR file to predict."
        path = file.name if hasattr(file, "name") else file
        try:
            atoms = read_structure(path)
            r = predict_structure(atoms, checkpoints)
        except Exception as e:  # surface parse/inference errors in the UI, never crash
            return f"Error: {e}"
        return (
            f"P(stable) = {r['p_stable']:.3f} +/- {r['p_stable_std']:.3f}\n"
            f"Energy/atom = {r['energy_pred']:.3f} +/- {r['energy_std']:.3f} eV"
        )

    def on_pick(gid) -> str:
        try:
            gid_int = int(gid) if gid is not None else None
        except (TypeError, ValueError):
            return "Pick a known MOF id."
        if gid_int is None or gid_int not in scored.index:
            return "Pick a known MOF id."
        row = scored.loc[gid_int]
        asa = row["geom_asa_m2_per_g"]
        asa_str = "N/A" if pd.isna(asa) else f"{asa:.0f}"
        return (
            f"P(stable) = {row['p_stable']:.3f} +/- {row['p_stable_std']:.3f}\n"
            f"Energy/atom = {row['energy_pred']:.3f} +/- {row['energy_std']:.3f} eV\n"
            f"Carbon-capture score = {row['carbon_score']:.3f} "
            f"(PLD {row['geom_pld']:.2f} A, ASA {asa_str} m2/g)"
        )

    with gr.Blocks(title="OrbScreen", theme="soft") as demo:
        gr.Markdown("# OrbScreen - fast MOF stability screening")
        gr.Markdown(_DESC)
        with gr.Tab("Predict"):
            with gr.Row():
                with gr.Column():
                    up = gr.File(label="Upload CIF / POSCAR", file_types=[".cif", ".vasp", ".poscar"])
                    up_btn = gr.Button("Predict uploaded structure", variant="primary")
                with gr.Column():
                    pick = gr.Dropdown(choices=ids, label="...or pick a screened MOF id")
                    pick_btn = gr.Button("Show screened prediction")
            out = gr.Textbox(label="Prediction", lines=4)
            up_btn.click(on_upload, inputs=up, outputs=out)
            pick_btn.click(on_pick, inputs=pick, outputs=out)
        with gr.Tab("Carbon-capture leaderboard"):
            gr.Markdown("Top screened MOFs by carbon-capture proxy score.")
            gr.Dataframe(value=leaderboard, label="Top 50")
    return demo
