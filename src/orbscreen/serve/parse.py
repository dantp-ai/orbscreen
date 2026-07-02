"""Parse an uploaded structure file (CIF/POSCAR/...) into an ASE Atoms, with friendly errors."""

from ase import Atoms
from ase.io import read


def read_structure(path: str) -> Atoms:
    """Read a structure file via ASE; raise ValueError on anything malformed or empty."""
    try:
        atoms = read(path)
    except Exception as e:  # ASE raises many parser-specific exceptions
        raise ValueError(f"Could not parse structure file: {e}") from e
    if isinstance(atoms, list):  # multi-image files -> take the first frame
        atoms = atoms[0] if atoms else None
    if not isinstance(atoms, Atoms) or len(atoms) == 0:
        raise ValueError("Parsed structure is empty or invalid.")
    return atoms
