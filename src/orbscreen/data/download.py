from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from orbscreen import config


def list_files() -> list[str]:
    """All files in the MofasaDB dataset repo."""
    return list_repo_files(config.REPO_ID, repo_type="dataset")


def ensure_files(files: list[str]) -> dict[str, Path]:
    """Download the named files from MofasaDB into DATA_DIR; return local paths.

    Uses the HF cache; re-running is a no-op once files are present.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for f in files:
        local = hf_hub_download(
            repo_id=config.REPO_ID,
            filename=f,
            repo_type="dataset",
            local_dir=str(config.DATA_DIR),
        )
        out[f] = Path(local)
    return out
