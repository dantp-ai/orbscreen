"""Export a few in-distribution MOF structures from MofasaDB to CIF files.

These CIFs are handy demo material for the Gradio app's upload path: they are drawn from the
same corpus the surrogate was trained on, so predictions are reliable and reproducible, and
each file's id also exists in the leaderboard / corpus-pick path for a side-by-side check.

Usage (from the repo root):
    uv run python scripts/export_demo_cifs.py                 # 3 CIFs -> ./demo_cifs/
    uv run python scripts/export_demo_cifs.py --n 5 --out demo_cifs
"""

import argparse
from pathlib import Path

from ase.db import connect
from ase.io import write

from orbscreen.data import download


def export_demo_cifs(n: int, out_dir: Path) -> list[Path]:
    """Write the first `n` corpus structures to CIFs in `out_dir`; return the file paths.

    Resolves `samples.db` via the project downloader: it uses the local copy in `data/` if
    present, otherwise fetches it from MofasaDB first.
    """
    samples_db = download.ensure_files(["samples.db"])["samples.db"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in connect(str(samples_db)).select(limit=n):
        cif = out_dir / f"demo_mof_{row.id}.cif"
        write(str(cif), row.toatoms())
        paths.append(cif)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export sample MOF CIFs for the OrbScreen demo upload path."
    )
    ap.add_argument("--n", type=int, default=3, help="number of structures to export (default 3)")
    ap.add_argument("--out", default="demo_cifs", help="output directory (default ./demo_cifs)")
    args = ap.parse_args()

    paths = export_demo_cifs(args.n, Path(args.out))
    print(f"wrote {len(paths)} CIF(s) to {args.out}/:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
