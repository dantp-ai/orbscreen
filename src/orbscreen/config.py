from pathlib import Path

REPO_ID = "Orbital-Materials/MofasaDB"
DATA_DIR = Path("data")
UNRELAXED_DB = "samples.db"
RELAXED_DB = "relaxed.db"

# pyzeo geometric descriptors parsed as model features (stored as strings / 'None').
GEOMETRY_KEYS = [
    "lcd",
    "pld",
    "dif",
    "number_of_channels",
    "number_of_pockets",
    "av_volume_fraction",
    "av_cm3_per_g",
    "nav_volume_fraction",
    "nav_cm3_per_g",
    "channel_volume_fraction",
    "pocket_volume_fraction",
    "asa_m2_per_cm3",
    "asa_m2_per_g",
    "nasa_m2_per_cm3",
    "nasa_m2_per_g",
]

# Topology codes that are not real RCSR frameworks (excluded from the topology split).
INVALID_TOPOLOGIES = {"ERROR", "UNKNOWN", "NA", "", None}
