"""Target definitions for OrbScreen.

- Regression target: ``energy_per_atom`` (the Orb-v3 relaxed energy/atom) is taken
  directly from the relaxed row (see parse.relaxed_fields).
- Stability label: a composite of MofasaDB validity flags plus relaxation
  convergence. Sensitivity = which flags compose it (see docs/data-schema.md).
"""

STABILITY_FLAGS = ("mofchecker_valid", "smact_valid", "no_atom_too_close", "geo_converged")


def stability_label(rec: dict) -> int:
    """1 iff all validity flags pass, relaxation converged, and reconstruction did not fail."""
    passes = all(bool(rec.get(flag)) for flag in STABILITY_FLAGS)
    return int(passes and not bool(rec.get("reconstruction_failed")))
