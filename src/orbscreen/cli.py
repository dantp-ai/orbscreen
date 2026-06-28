"""Command-line interface for OrbScreen.

Commands:
  orbscreen build   -- Download MofasaDB files (already cached), build and
                       write the unified Parquet dataset.
  orbscreen baseline -- Read a Parquet dataset and run the descriptor baseline
                        for both random and topology splits, printing JSON metrics.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from orbscreen import config
from orbscreen.data import download
from orbscreen.data.build import build_dataset
from orbscreen.models.baseline import train_baseline


def _build(args) -> None:
    paths = download.ensure_files([config.UNRELAXED_DB, config.RELAXED_DB])
    limit = args.limit if hasattr(args, "limit") else None
    df, stats = build_dataset(
        paths[config.UNRELAXED_DB],
        paths[config.RELAXED_DB],
        args.out,
        limit=limit,
    )
    print(f"build stats: {json.dumps(stats)}")
    print(f"rows written: {len(df)}")
    print(f"output: {args.out}")


def _baseline(args) -> None:
    df = pd.read_parquet(args.data)
    results: dict = {}
    for split in ["split_random", "split_topology"]:
        results[split] = train_baseline(df, split_col=split)
    text = json.dumps(results, indent=2)
    print(text)
    if getattr(args, "out", None):
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")


def _cascade(args) -> None:
    from orbscreen.screen.report import run_cascade_analysis

    res = run_cascade_analysis(
        args.predictions, args.screen_timing, args.orb_benchmark, args.out_dir,
    )
    print(json.dumps(res["headline"], indent=2))
    print(f"wrote {args.out_dir}/results_screen.json, cascade.json, cascade_*.png")


def main() -> None:
    p = argparse.ArgumentParser(prog="orbscreen", description="OrbScreen Phase 1 CLI")
    sub = p.add_subparsers(required=True, dest="command")

    b = sub.add_parser("build", help="Build the unified Parquet dataset")
    b.add_argument("--out", default="data/dataset.parquet", help="Output Parquet path")
    b.add_argument("--limit", type=int, default=None, help="Cap rows (for quick tests)")
    b.set_defaults(func=_build)

    m = sub.add_parser("baseline", help="Run descriptor baseline and print metrics")
    m.add_argument("--data", default="data/dataset.parquet", help="Input Parquet path")
    m.add_argument("--out", default=None, help="Optional path to write metrics JSON")
    m.set_defaults(func=_baseline)

    c = sub.add_parser("cascade", help="Cost-accuracy cascade analysis from screen artifacts")
    c.add_argument("--predictions", required=True, help="screen_predictions.parquet")
    c.add_argument("--screen-timing", dest="screen_timing", required=True,
                   help="screen_timing.json (surrogate throughput)")
    c.add_argument("--orb-benchmark", dest="orb_benchmark", required=True,
                   help="benchmark_orb.json (Orb-v3 throughput)")
    c.add_argument("--out-dir", dest="out_dir", default=".", help="output directory")
    c.set_defaults(func=_cascade)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
