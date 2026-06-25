"""Command-line interface for OrbScreen.

Commands:
  orbscreen build   -- Download MofasaDB files (already cached), build and
                       write the unified Parquet dataset.
  orbscreen baseline -- Read a Parquet dataset and run the descriptor baseline
                        for both random and topology splits, printing JSON metrics.
"""

import argparse
import json

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
    print(json.dumps(results, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="orbscreen", description="OrbScreen Phase 1 CLI")
    sub = p.add_subparsers(required=True, dest="command")

    b = sub.add_parser("build", help="Build the unified Parquet dataset")
    b.add_argument("--out", default="data/dataset.parquet", help="Output Parquet path")
    b.add_argument("--limit", type=int, default=None, help="Cap rows (for quick tests)")
    b.set_defaults(func=_build)

    m = sub.add_parser("baseline", help="Run descriptor baseline and print metrics")
    m.add_argument("--data", default="data/dataset.parquet", help="Input Parquet path")
    m.set_defaults(func=_baseline)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
