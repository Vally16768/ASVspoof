#paths.py
from __future__ import annotations
from pathlib import Path
from typing import Dict

def protocol_paths(root: Path) -> Dict[str, Path]:
    """Return paths to CM protocol files for each split."""
    cm_dir = root / "ASVspoof2019_LA_cm_protocols"
    return {
        "train": cm_dir / "ASVspoof2019.LA.cm.train.trn.txt",
        "dev":   cm_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval":  cm_dir / "ASVspoof2019.LA.cm.eval.trl.txt",  # unlabeled
    }

def audio_dir_for_split(root: Path, split: str) -> Path:
    """Return the FLAC directory for a given split."""
    if split == "train":
        return root / "ASVspoof2019_LA_train" / "flac"
    if split == "dev":
        return root / "ASVspoof2019_LA_dev" / "flac"
    if split == "eval":
        return root / "ASVspoof2019_LA_eval" / "flac"
    raise ValueError(f"Unknown split: {split}")
